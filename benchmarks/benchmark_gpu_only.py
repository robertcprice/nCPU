#!/usr/bin/env python3
"""GPU execution benchmark for representative `run_gpu_only()` workloads.

Measures the throughput of the NeuralCPU `run_gpu_only()` path and, when the
local Rust/Metal extension is available, compares it against Rust/Metal
backends on the same ARM64 program.

Supported workloads are selected with `NCPU_GPU_BENCH_WORKLOAD`:
  - `counted`       : arithmetic loop closed by `B.NE`
  - `adjacent-counted`: two adjacent counted loops to exercise chained handoff
  - `counted-bytecopy`: counted loop followed by a byte-copy loop
  - `bytecopy`      : byte-copy loop closed by `B.NE`
  - `bytecopy-counted`: byte-copy loop followed by a counted loop
  - `adjacent-bytecopy`: two adjacent byte-copy loops closed by `B.NE`
  - `bytecopy-cbnz` : byte-copy loop closed by `CBNZ`
  - `bytecopy-cbz-exit`: byte-copy loop with top-of-loop `CBZ`, backedge `B`
  - `bytecopy-bge-exit`: byte-copy loop with top-of-loop `CMP` + `B.GE`, backedge `B`
  - `bytecopy-cbz-then-bge-exit`: top-exit `CBZ` byte-copy loop followed by top-exit `B.GE`
  - `adjacent-bytecopy-bge-exit`: two adjacent top-exit compare byte-copy loops
  - `bytecopy-blt`  : byte-copy loop closed by `CMP` + `B.LT`
  - `nested-counted`: two-level arithmetic loop with nested `B.NE`

Usage:
    python benchmarks/benchmark_gpu_only.py

Typical output is hardware- and model-set-dependent. Use this benchmark for
relative comparisons on the same machine; do not treat the IPS as a portable
published constant across branches or devices.
"""

import argparse
import importlib.util
import io
import json
import os
import struct
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────

N_ITERS      = 2000          # loop iteration count
NESTED_OUTER = int(os.environ.get("NCPU_GPU_BENCH_NESTED_OUTER", "128"))
NESTED_INNER = int(os.environ.get("NCPU_GPU_BENCH_NESTED_INNER", "128"))
LOAD_ADDR    = 0x10000       # same load address used by other benchmarks
N_RUNS       = 3             # timed measurement repetitions
RUST_RUNS    = 3             # timed repetitions for Rust/Metal comparison
MAX_INSTS    = 200_000       # upper bound; program halts well before this
BATCH_SIZE   = 64            # matches run_gpu_only() default
EXPERIMENTAL_RUST = os.environ.get("NCPU_BENCH_EXPERIMENTAL_RUST", "0") == "1"
WORKLOAD = os.environ.get("NCPU_GPU_BENCH_WORKLOAD", "counted").strip().lower()
BYTECOPY_SRC_ADDR = 0x2000
BYTECOPY_DST_ADDR = 0x3000
BYTECOPY_SECOND_SRC_ADDR = 0x2800
BYTECOPY_SECOND_DST_ADDR = 0x3800
BENCHMARK_WORKLOADS = (
    "counted",
    "adjacent-counted",
    "counted-bytecopy",
    "bytecopy",
    "bytecopy-counted",
    "adjacent-bytecopy",
    "bytecopy-cbnz",
    "bytecopy-cbz-exit",
    "bytecopy-bge-exit",
    "bytecopy-cbz-then-bge-exit",
    "adjacent-bytecopy-bge-exit",
    "bytecopy-blt",
    "nested-counted",
)

# ── ARM64 program builders ────────────────────────────────────────────────────

def movz_x(rd: int, imm16: int) -> int:
    return 0xD2800000 | ((imm16 & 0xFFFF) << 5) | rd


def add_x_imm(rd: int, rn: int, imm12: int) -> int:
    return 0x91000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd


def subs_x_imm(rd: int, rn: int, imm12: int) -> int:
    return 0xF1000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd


def ldrb_uoff(rt: int, rn: int, imm12: int = 0) -> int:
    return 0x39400000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rt


def strb_uoff(rt: int, rn: int, imm12: int = 0) -> int:
    return 0x39000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rt


def b_ne(offset_words: int) -> int:
    return 0x54000000 | ((offset_words & 0x7FFFF) << 5) | 0x1


def b_uncond(offset_words: int) -> int:
    return 0x14000000 | (offset_words & 0x3FFFFFF)


def cbz(rt: int, offset_words: int) -> int:
    return 0xB4000000 | ((offset_words & 0x7FFFF) << 5) | rt


def cmp_reg(rn: int, rm: int) -> int:
    return 0xEB000000 | (rm << 16) | (rn << 5) | 31


def _concat_programs(*programs: bytes) -> bytes:
    """Concatenate loop snippets, dropping intermediate HALTs."""
    words: list[int] = []
    for idx, program in enumerate(programs):
        chunk = list(struct.unpack(f"<{len(program) // 4}I", program))
        if idx < len(programs) - 1:
            if not chunk or chunk[-1] != 0x00000000:
                raise ValueError("expected intermediate program chunk to end with HALT")
            chunk = chunk[:-1]
        words.extend(chunk)
    return struct.pack(f"<{len(words)}I", *words)


def build_program(n: int) -> bytes:
    """Return raw ARM64 bytes for the tight add-loop.

    Encoding reference (AArch64 A64 ISA):
      MOVZ Xd, #imm16 : 0xD2800000 | (imm16 << 5) | Rd
      ADD  Xd, Xn, Xm : 0x8B000000 | (Rm << 16) | (Rn << 5) | Rd
      SUBS Xd, Xn, #imm12 : 0xF1000000 | (imm12 << 10) | (Rn << 5) | Rd
      B.cond #imm19   : 0x54000000 | (imm19 << 5) | cond
        cond NE = 0x1
      HALT            : 0x00000000
    """
    insts: list[int] = []

    # MOVZ X0, #0
    insts.append(0xD2800000 | (0 << 5) | 0)

    # MOVZ X1, #n  (counter)
    insts.append(0xD2800000 | ((n & 0xFFFF) << 5) | 1)

    # MOVZ X2, #1  (step)
    insts.append(0xD2800000 | (1 << 5) | 2)

    # --- loop body starts here (byte offset = len(insts)*4 = 12) ---
    loop_byte_offset = len(insts) * 4

    # ADD X0, X0, X2  (acc += step)
    insts.append(0x8B000000 | (2 << 16) | (0 << 5) | 0)

    # SUBS X1, X1, #1  (counter--; update flags)
    insts.append(0xF1000000 | (1 << 10) | (1 << 5) | 1)

    # B.NE loop  (branch back by 2 instructions = -8 bytes)
    branch_byte_offset = loop_byte_offset - len(insts) * 4  # negative
    imm19 = (branch_byte_offset >> 2) & 0x7FFFF             # signed 19-bit
    insts.append(0x54000000 | (imm19 << 5) | 0x1)           # cond NE = 1

    # HALT
    insts.append(0x00000000)

    return struct.pack(f"<{len(insts)}I", *insts)


def build_adjacent_counted_program(first_n: int, second_n: int | None = None) -> bytes:
    """Return raw ARM64 bytes for two adjacent counted loops."""
    if second_n is None:
        second_n = first_n

    insts = [
        movz_x(0, 0),
        movz_x(1, first_n),
        movz_x(2, 1),
        0x8B000000 | (2 << 16) | (0 << 5) | 0,
        0xF1000000 | (1 << 10) | (1 << 5) | 1,
        b_ne((-2) & 0x7FFFF),
        movz_x(4, 0),
        movz_x(5, second_n),
        movz_x(6, 1),
        0x8B000000 | (6 << 16) | (4 << 5) | 4,
        0xF1000000 | (1 << 10) | (5 << 5) | 5,
        b_ne((-2) & 0x7FFFF),
        0x00000000,
    ]
    return struct.pack(f"<{len(insts)}I", *insts)


def build_counted_then_bytecopy_program(
    first_n: int,
    second_n: int | None = None,
    *,
    src_addr: int = BYTECOPY_SRC_ADDR,
    dst_addr: int = BYTECOPY_DST_ADDR,
) -> bytes:
    """Return raw ARM64 bytes for a counted loop followed by a byte-copy loop."""
    if second_n is None:
        second_n = first_n
    return _concat_programs(
        build_program(first_n),
        build_bytecopy_program(second_n, src_addr, dst_addr),
    )


def build_bytecopy_program(n: int,
                           src_addr: int = BYTECOPY_SRC_ADDR,
                           dst_addr: int = BYTECOPY_DST_ADDR) -> bytes:
    """Return raw ARM64 bytes for a simple byte-copy loop."""
    insts = [
        movz_x(1, src_addr),
        movz_x(3, dst_addr),
        movz_x(2, n),
        ldrb_uoff(4, 1, 0),
        strb_uoff(4, 3, 0),
        add_x_imm(1, 1, 1),
        add_x_imm(3, 3, 1),
        subs_x_imm(2, 2, 1),
        b_ne((-5) & 0x7FFFF),
        0x00000000,
    ]
    return struct.pack(f"<{len(insts)}I", *insts)


def build_adjacent_bytecopy_program(
    first_n: int,
    second_n: int | None = None,
    *,
    first_src_addr: int = BYTECOPY_SRC_ADDR,
    first_dst_addr: int = BYTECOPY_DST_ADDR,
    second_src_addr: int = BYTECOPY_SECOND_SRC_ADDR,
    second_dst_addr: int = BYTECOPY_SECOND_DST_ADDR,
) -> bytes:
    """Return raw ARM64 bytes for two adjacent byte-copy loops."""
    if second_n is None:
        second_n = first_n
    return _concat_programs(
        build_bytecopy_program(first_n, first_src_addr, first_dst_addr),
        build_bytecopy_program(second_n, second_src_addr, second_dst_addr),
    )


def build_bytecopy_then_counted_program(
    first_n: int,
    second_n: int | None = None,
    *,
    src_addr: int = BYTECOPY_SRC_ADDR,
    dst_addr: int = BYTECOPY_DST_ADDR,
) -> bytes:
    """Return raw ARM64 bytes for a byte-copy loop followed by a counted loop."""
    if second_n is None:
        second_n = first_n
    return _concat_programs(
        build_bytecopy_program(first_n, src_addr, dst_addr),
        build_program(second_n),
    )


def build_bytecopy_cbnz_program(n: int,
                                src_addr: int = BYTECOPY_SRC_ADDR,
                                dst_addr: int = BYTECOPY_DST_ADDR) -> bytes:
    """Return raw ARM64 bytes for a byte-copy loop closed by CBNZ."""
    insts = [
        movz_x(1, src_addr),
        movz_x(3, dst_addr),
        movz_x(2, n),
        ldrb_uoff(4, 1, 0),
        strb_uoff(4, 3, 0),
        add_x_imm(1, 1, 1),
        add_x_imm(3, 3, 1),
        subs_x_imm(2, 2, 1),
        0xB5000000 | (((-5) & 0x7FFFF) << 5) | 2,  # CBNZ X2, loop
        0x00000000,
    ]
    return struct.pack(f"<{len(insts)}I", *insts)


def build_bytecopy_cbz_exit_program(n: int,
                                    src_addr: int = BYTECOPY_SRC_ADDR,
                                    dst_addr: int = BYTECOPY_DST_ADDR) -> bytes:
    """Return raw ARM64 bytes for a top-exit byte-copy loop."""
    insts = [
        movz_x(1, src_addr),
        movz_x(3, dst_addr),
        movz_x(2, n),
        cbz(2, 7),                 # CBZ X2, halt
        ldrb_uoff(4, 1, 0),
        strb_uoff(4, 3, 0),
        add_x_imm(1, 1, 1),
        add_x_imm(3, 3, 1),
        subs_x_imm(2, 2, 1),
        b_uncond((-6) & 0x3FFFFFF),  # B loop_head
        0x00000000,
    ]
    return struct.pack(f"<{len(insts)}I", *insts)


def build_bytecopy_bge_exit_program(n: int,
                                    src_addr: int = BYTECOPY_SRC_ADDR,
                                    dst_addr: int = BYTECOPY_DST_ADDR) -> bytes:
    """Return raw ARM64 bytes for a top-exit compare byte-copy loop."""
    insts = [
        movz_x(1, src_addr),
        movz_x(3, dst_addr),
        movz_x(2, 0),
        movz_x(5, n),
        cmp_reg(2, 5),
        0x54000000 | ((7 & 0x7FFFF) << 5) | 0xA,  # B.GE halt
        ldrb_uoff(4, 1, 0),
        strb_uoff(4, 3, 0),
        add_x_imm(1, 1, 1),
        add_x_imm(3, 3, 1),
        add_x_imm(2, 2, 1),
        b_uncond((-7) & 0x3FFFFFF),  # B loop_head (CMP)
        0x00000000,
    ]
    return struct.pack(f"<{len(insts)}I", *insts)


def build_adjacent_bytecopy_bge_exit_program(
    first_n: int,
    second_n: int | None = None,
    *,
    first_src_addr: int = BYTECOPY_SRC_ADDR,
    first_dst_addr: int = BYTECOPY_DST_ADDR,
    second_src_addr: int = BYTECOPY_SECOND_SRC_ADDR,
    second_dst_addr: int = BYTECOPY_SECOND_DST_ADDR,
) -> bytes:
    """Return raw ARM64 bytes for two adjacent top-exit compare byte-copy loops."""
    if second_n is None:
        second_n = first_n
    return _concat_programs(
        build_bytecopy_bge_exit_program(first_n, first_src_addr, first_dst_addr),
        build_bytecopy_bge_exit_program(second_n, second_src_addr, second_dst_addr),
    )


def build_bytecopy_cbz_then_bge_exit_program(
    first_n: int,
    second_n: int | None = None,
    *,
    first_src_addr: int = BYTECOPY_SRC_ADDR,
    first_dst_addr: int = BYTECOPY_DST_ADDR,
    second_src_addr: int = BYTECOPY_SECOND_SRC_ADDR,
    second_dst_addr: int = BYTECOPY_SECOND_DST_ADDR,
) -> bytes:
    """Return raw ARM64 bytes for mixed top-exit byte-copy loops."""
    if second_n is None:
        second_n = first_n
    return _concat_programs(
        build_bytecopy_cbz_exit_program(first_n, first_src_addr, first_dst_addr),
        build_bytecopy_bge_exit_program(second_n, second_src_addr, second_dst_addr),
    )


def build_bytecopy_blt_program(n: int,
                               src_addr: int = BYTECOPY_SRC_ADDR,
                               dst_addr: int = BYTECOPY_DST_ADDR) -> bytes:
    """Return raw ARM64 bytes for a byte-copy loop closed by CMP + B.LT."""
    insts = [
        movz_x(1, src_addr),
        movz_x(3, dst_addr),
        movz_x(2, 0),
        movz_x(5, n),
        ldrb_uoff(4, 1, 0),
        strb_uoff(4, 3, 0),
        add_x_imm(1, 1, 1),
        add_x_imm(3, 3, 1),
        add_x_imm(2, 2, 1),
        cmp_reg(2, 5),
        0x54000000 | (((-6) & 0x7FFFF) << 5) | 0xB,  # B.LT loop
        0x00000000,
    ]
    return struct.pack(f"<{len(insts)}I", *insts)


def build_nested_counted_program(outer_n: int = NESTED_OUTER,
                                 inner_n: int = NESTED_INNER) -> bytes:
    """Return raw ARM64 bytes for a simple nested counted loop."""
    insts = [
        movz_x(0, outer_n),
        movz_x(2, 0),
        movz_x(1, inner_n),
        add_x_imm(2, 2, 1),
        subs_x_imm(1, 1, 1),
        b_ne((-2) & 0x7FFFF),
        subs_x_imm(0, 0, 1),
        b_ne((-5) & 0x7FFFF),
        0x00000000,
    ]
    return struct.pack(f"<{len(insts)}I", *insts)


def architectural_instruction_count(n_iters: int, workload: str = "counted") -> int:
    """Architectural dynamic instruction count for the benchmark program."""
    if workload == "adjacent-counted":
        return (6 * n_iters) + 7
    if workload in {"counted-bytecopy", "bytecopy-counted"}:
        return (9 * n_iters) + 7
    if workload == "adjacent-bytecopy":
        return (12 * n_iters) + 7
    if workload in {"bytecopy", "bytecopy-cbnz"}:
        return 3 + n_iters * 6 + 1
    if workload == "bytecopy-cbz-exit":
        return 3 + (7 * n_iters) + 2
    if workload == "bytecopy-bge-exit":
        return 4 + (8 * n_iters) + 3
    if workload == "bytecopy-cbz-then-bge-exit":
        return (15 * n_iters) + 11
    if workload == "adjacent-bytecopy-bge-exit":
        return (16 * n_iters) + 13
    if workload == "bytecopy-blt":
        return 4 + n_iters * 7 + 1
    if workload == "nested-counted":
        return 3 + NESTED_OUTER * ((3 * NESTED_INNER) + 3)
    return 3 + n_iters * 3 + 1


def expected_engine_executed_count(n_iters: int, workload: str = "counted") -> int:
    """Expected `run_gpu_only()` executed-count for the benchmark program.

    `run_gpu_only()` does not report a raw retired-instruction count. It counts
    the loop body and setup work, but stop instructions such as `HALT` and the
    terminal loop-exit branch are not included in the final executed-count.
    """
    if workload == "adjacent-counted":
        return (6 * max(n_iters, 0)) + 4
    if workload in {"counted-bytecopy", "bytecopy-counted"}:
        return (9 * max(n_iters, 0)) + 4
    if workload == "adjacent-bytecopy":
        return (12 * max(n_iters, 0)) + 4
    if workload == "bytecopy-cbz-exit":
        return 3 + (7 * max(n_iters, 0))
    if workload == "bytecopy-bge-exit":
        return 4 + (8 * max(n_iters, 0)) + 1
    if workload == "bytecopy-cbz-then-bge-exit":
        return (15 * max(n_iters, 0)) + 8
    if workload == "adjacent-bytecopy-bge-exit":
        return (16 * max(n_iters, 0)) + 10
    if n_iters <= 0:
        return 2
    if workload in {"bytecopy", "bytecopy-cbnz"}:
        return 3 + (5 * n_iters) + (n_iters - 1)
    if workload == "bytecopy-blt":
        return 4 + (6 * n_iters) + (n_iters - 1)
    if workload == "nested-counted":
        return 1 + NESTED_OUTER * ((3 * NESTED_INNER) + 3)
    return 3 + (2 * n_iters) + (n_iters - 1)


def make_bytecopy_payload(n_iters: int, *, seed: int = 3) -> bytes:
    return bytes((i * 17 + seed) & 0xFF for i in range(n_iters))


def load_ncpu_metal():
    """Load the Rust/Metal extension from the local tree or an installed package.

    The local tree path is preferred so an in-tree build overrides a stale
    site-packages install. If the .so isn't present locally, fall back to
    ``import ncpu_metal`` so ``maturin develop`` / ``pip install`` layouts
    (common in CI) work without copying files into the source tree.
    """
    so_path = PROJECT_ROOT / "kernels" / "rust_metal" / "ncpu_metal.abi3.so"
    if so_path.exists():
        spec = importlib.util.spec_from_file_location("ncpu_metal", str(so_path))
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    try:
        import ncpu_metal  # type: ignore[import-not-found]
    except ImportError:
        return None
    return ncpu_metal


# ── CPU helpers ───────────────────────────────────────────────────────────────

def load_program(cpu, code: bytes, load_addr: int) -> None:
    """Write raw bytes into the CPU memory tensor and reset execution state."""
    import torch
    for i, byte_val in enumerate(code):
        cpu.memory[load_addr + i] = byte_val
    cpu.pc    = torch.tensor(load_addr, dtype=torch.int64, device=cpu.device)
    cpu.regs[:] = 0
    cpu.flags[:] = 0
    cpu.halted  = False


def reset_pc(cpu, load_addr: int) -> None:
    """Reset PC and registers for a repeat run without reloading the program."""
    import torch
    cpu.pc    = torch.tensor(load_addr, dtype=torch.int64, device=cpu.device)
    cpu.regs[:] = 0
    cpu.flags[:] = 0
    cpu.halted  = False


def benchmark_rust_backend(mod, cls_name: str, code: bytes, load_addr: int, arch_insts: int,
                           setup_fn=None, verify_fn=None):
    """Benchmark one Rust/Metal backend using architectural IPS for comparison."""
    if mod is None or not hasattr(mod, cls_name):
        return None

    cls = getattr(mod, cls_name)
    if cls_name == "FullARM64CPU":
        cpu = cls(memory_size=4 * 1024 * 1024, cycles_per_batch=10_000_000)
        cpu.reset()
        cpu.load_program(list(code), load_addr)
        if setup_fn is not None:
            setup_fn(cpu)
        cpu.set_pc(load_addr)
        cpu.execute(max_cycles=MAX_INSTS)  # warm-up

        runs = []
        for _ in range(RUST_RUNS):
            cpu.reset()
            cpu.load_program(list(code), load_addr)
            if setup_fn is not None:
                setup_fn(cpu)
            cpu.set_pc(load_addr)
            result = cpu.execute(max_cycles=MAX_INSTS)
            runs.append((result.elapsed_seconds, cpu.get_register(0), result.total_cycles, verify_fn(cpu) if verify_fn else True))
    else:
        cpu = cls(memory_size=4 * 1024 * 1024)
        cpu.reset()
        cpu.load_program(list(code), load_addr)
        if setup_fn is not None:
            setup_fn(cpu)
        cpu.set_pc(load_addr)
        cpu.execute(max_batches=100, timeout_seconds=10.0)  # warm-up

        runs = []
        for _ in range(RUST_RUNS):
            cpu.reset()
            cpu.load_program(list(code), load_addr)
            if setup_fn is not None:
                setup_fn(cpu)
            cpu.set_pc(load_addr)
            result = cpu.execute(max_batches=100, timeout_seconds=10.0)
            runs.append((result.elapsed_seconds, cpu.get_register(0), result.total_cycles, verify_fn(cpu) if verify_fn else True))

    arch_ips = [arch_insts / elapsed if elapsed > 0 else 0.0 for elapsed, _, _, _ in runs]
    return {
        "name": cls_name,
        "avg_ips": sum(arch_ips) / len(arch_ips),
        "peak_ips": max(arch_ips),
        "min_ips": min(arch_ips),
        "native_cycles": runs[-1][2],
        "x0_ok": all(x0 == N_ITERS for _, x0, _, _ in runs) if verify_fn is None else True,
        "verify_ok": all(ok for _, _, _, ok in runs),
    }


# ── Main benchmark ────────────────────────────────────────────────────────────

def benchmark(
    workload: str | None = None,
    *,
    compare_rust: bool = True,
    require_backend_prefix: str | None = None,
    cpu: Any = None,
) -> dict:
    global WORKLOAD
    prev_workload = WORKLOAD
    if workload is not None:
        WORKLOAD = workload
    print()
    print("nCPU GPU Execution Benchmark")
    print("=" * 50)
    if WORKLOAD == "adjacent-counted":
        print(f"Program: two adjacent counted loops, N={N_ITERS} iterations each")
        print(f"Expected result: X0 = {N_ITERS}, X4 = {N_ITERS}")
        print("Note: executed-count excludes HALT and both loops' final untaken exit branches.")
    elif WORKLOAD == "counted-bytecopy":
        print(f"Program: counted loop followed by LDRB+STRB byte-copy loop, N={N_ITERS} iterations each")
        print(
            f"Expected result: X0 = {N_ITERS}, and {N_ITERS} bytes copied "
            f"from 0x{BYTECOPY_SRC_ADDR:X} to 0x{BYTECOPY_DST_ADDR:X}"
        )
        print("Note: executed-count excludes HALT and each loop's final untaken exit branch.")
    elif WORKLOAD == "adjacent-bytecopy":
        print(f"Program: two adjacent LDRB+STRB byte-copy loops (B.NE), N={N_ITERS} iterations each")
        print(
            f"Expected result: {N_ITERS} bytes copied "
            f"from 0x{BYTECOPY_SRC_ADDR:X}->0x{BYTECOPY_DST_ADDR:X} and "
            f"0x{BYTECOPY_SECOND_SRC_ADDR:X}->0x{BYTECOPY_SECOND_DST_ADDR:X}"
        )
        print("Note: executed-count excludes HALT and both loops' final untaken exit branches.")
    elif WORKLOAD == "bytecopy-counted":
        print(f"Program: LDRB+STRB byte-copy loop followed by a counted loop, N={N_ITERS} iterations each")
        print(
            f"Expected result: {N_ITERS} bytes copied from 0x{BYTECOPY_SRC_ADDR:X} to 0x{BYTECOPY_DST_ADDR:X}, "
            f"then X0 = {N_ITERS}"
        )
        print("Note: executed-count excludes HALT and each loop's final untaken exit branch.")
    elif WORKLOAD in {"bytecopy", "bytecopy-cbnz", "bytecopy-cbz-exit", "bytecopy-bge-exit", "bytecopy-blt"}:
        if WORKLOAD == "bytecopy-cbnz":
            branch_name = "CBNZ"
        elif WORKLOAD == "bytecopy-cbz-exit":
            branch_name = "CBZ + B"
        elif WORKLOAD == "bytecopy-bge-exit":
            branch_name = "CMP + B.GE + B"
        elif WORKLOAD == "bytecopy-blt":
            branch_name = "CMP + B.LT"
        else:
            branch_name = "B.NE"
        print(f"Program: LDRB+STRB byte-copy loop ({branch_name}), N={N_ITERS} iterations")
        print(
            f"Expected result: {N_ITERS} bytes copied "
            f"from 0x{BYTECOPY_SRC_ADDR:X} to 0x{BYTECOPY_DST_ADDR:X}"
        )
        if WORKLOAD in {"bytecopy-cbz-exit", "bytecopy-bge-exit"}:
            print("Note: executed-count excludes HALT and the final taken loop-exit branch.")
        else:
            print("Note: executed-count excludes HALT and the final untaken loop-exit branch.")
    elif WORKLOAD == "bytecopy-cbz-then-bge-exit":
        print(f"Program: top-exit CBZ byte-copy loop followed by top-exit CMP+B.GE byte-copy loop, N={N_ITERS} iterations each")
        print(
            f"Expected result: {N_ITERS} bytes copied "
            f"from 0x{BYTECOPY_SRC_ADDR:X}->0x{BYTECOPY_DST_ADDR:X} and "
            f"0x{BYTECOPY_SECOND_SRC_ADDR:X}->0x{BYTECOPY_SECOND_DST_ADDR:X}"
        )
        print("Note: executed-count excludes HALT, the first loop's taken CBZ exit, and the second loop's taken B.GE exit.")
    elif WORKLOAD == "adjacent-bytecopy-bge-exit":
        print(f"Program: two adjacent LDRB+STRB byte-copy loops (CMP + B.GE + B), N={N_ITERS} iterations each")
        print(
            f"Expected result: {N_ITERS} bytes copied "
            f"from 0x{BYTECOPY_SRC_ADDR:X}->0x{BYTECOPY_DST_ADDR:X} and "
            f"0x{BYTECOPY_SECOND_SRC_ADDR:X}->0x{BYTECOPY_SECOND_DST_ADDR:X}"
        )
        print("Note: executed-count excludes HALT and both loops' final taken loop-exit branches.")
    elif WORKLOAD == "nested-counted":
        print(
            f"Program: nested counted loop, outer={NESTED_OUTER}, "
            f"inner={NESTED_INNER}"
        )
        print(
            f"Expected result: X2 = {NESTED_OUTER * NESTED_INNER} "
            f"(one increment per inner iteration)"
        )
        print("Note: executed-count excludes HALT and each loop's untaken exit branch.")
    else:
        print(f"Program: tight ADD+SUBS+B.NE loop, N={N_ITERS} iterations")
        print(f"Expected result: X0 = {N_ITERS} (each iteration adds step=1)")
        print("Note: executed-count excludes HALT and the final untaken loop-exit branch.")
    print(f"run_gpu_only() sync interval: {os.environ.get('NCPU_GPU_SYNC_INTERVAL', '2')}")
    print()

    # -- Import -----------------------------------------------------------------
    try:
        import torch
        from ncpu.neural.cpu import NeuralCPU
    except ImportError as exc:
        print(f"[ERROR] Could not import NeuralCPU: {exc}")
        print("        Make sure you are running from the project root.")
        raise

    # -- Build program ----------------------------------------------------------
    if WORKLOAD == "bytecopy":
        code = build_bytecopy_program(N_ITERS)
    elif WORKLOAD == "adjacent-counted":
        code = build_adjacent_counted_program(N_ITERS)
    elif WORKLOAD == "counted-bytecopy":
        code = build_counted_then_bytecopy_program(N_ITERS)
    elif WORKLOAD == "adjacent-bytecopy":
        code = build_adjacent_bytecopy_program(N_ITERS)
    elif WORKLOAD == "bytecopy-counted":
        code = build_bytecopy_then_counted_program(N_ITERS)
    elif WORKLOAD == "bytecopy-cbnz":
        code = build_bytecopy_cbnz_program(N_ITERS)
    elif WORKLOAD == "bytecopy-cbz-exit":
        code = build_bytecopy_cbz_exit_program(N_ITERS)
    elif WORKLOAD == "bytecopy-bge-exit":
        code = build_bytecopy_bge_exit_program(N_ITERS)
    elif WORKLOAD == "bytecopy-cbz-then-bge-exit":
        code = build_bytecopy_cbz_then_bge_exit_program(N_ITERS)
    elif WORKLOAD == "adjacent-bytecopy-bge-exit":
        code = build_adjacent_bytecopy_bge_exit_program(N_ITERS)
    elif WORKLOAD == "bytecopy-blt":
        code = build_bytecopy_blt_program(N_ITERS)
    elif WORKLOAD == "nested-counted":
        code = build_nested_counted_program()
    else:
        code = build_program(N_ITERS)
    n_program_insts = len(code) // 4
    arch_insts = architectural_instruction_count(N_ITERS, WORKLOAD)
    expected_insts = expected_engine_executed_count(N_ITERS, WORKLOAD)
    print(f"Program size: {n_program_insts} instructions ({len(code)} bytes)")
    print(f"Architectural instruction count: {arch_insts:,}")
    print(f"Expected run_gpu_only() executed count: {expected_insts:,}")
    print()

    # -- Create CPU (fast_mode=False uses neural ALU) ---------------------------
    if cpu is None:
        try:
            cpu = NeuralCPU(fast_mode=False)
        except Exception as exc:
            print(f"[ERROR] NeuralCPU construction failed: {exc}")
            raise

    # -- Load program once ------------------------------------------------------
    load_program(cpu, code, LOAD_ADDR)
    expected_copy_windows: list[tuple[int, int, bytes]] = []
    expected_x0 = None
    expected_x2 = None
    if WORKLOAD == "counted-bytecopy":
        expected_x0 = N_ITERS
        expected_x2 = 0
        expected_copy_windows = [
            (BYTECOPY_SRC_ADDR, BYTECOPY_DST_ADDR, make_bytecopy_payload(N_ITERS, seed=7)),
        ]
    elif WORKLOAD == "bytecopy-counted":
        expected_x0 = N_ITERS
        expected_x2 = 1
        expected_copy_windows = [
            (BYTECOPY_SRC_ADDR, BYTECOPY_DST_ADDR, make_bytecopy_payload(N_ITERS, seed=13)),
        ]
    if WORKLOAD in {"bytecopy", "bytecopy-cbnz", "bytecopy-cbz-exit", "bytecopy-bge-exit", "bytecopy-blt"}:
        expected_copy_windows = [
            (BYTECOPY_SRC_ADDR, BYTECOPY_DST_ADDR, make_bytecopy_payload(N_ITERS)),
        ]
    elif WORKLOAD == "adjacent-bytecopy":
        expected_copy_windows = [
            (BYTECOPY_SRC_ADDR, BYTECOPY_DST_ADDR, make_bytecopy_payload(N_ITERS, seed=3)),
            (BYTECOPY_SECOND_SRC_ADDR, BYTECOPY_SECOND_DST_ADDR, make_bytecopy_payload(N_ITERS, seed=11)),
        ]
    elif WORKLOAD == "bytecopy-cbz-then-bge-exit":
        expected_x2 = N_ITERS
        expected_copy_windows = [
            (BYTECOPY_SRC_ADDR, BYTECOPY_DST_ADDR, make_bytecopy_payload(N_ITERS, seed=23)),
            (BYTECOPY_SECOND_SRC_ADDR, BYTECOPY_SECOND_DST_ADDR, make_bytecopy_payload(N_ITERS, seed=29)),
        ]
    elif WORKLOAD == "adjacent-bytecopy-bge-exit":
        expected_copy_windows = [
            (BYTECOPY_SRC_ADDR, BYTECOPY_DST_ADDR, make_bytecopy_payload(N_ITERS, seed=5)),
            (BYTECOPY_SECOND_SRC_ADDR, BYTECOPY_SECOND_DST_ADDR, make_bytecopy_payload(N_ITERS, seed=19)),
        ]
    for src_addr, _dst_addr, payload in expected_copy_windows:
        cpu.memory[src_addr:src_addr + len(payload)] = torch.tensor(
            list(payload), dtype=torch.uint8, device=cpu.device
        )

    # -- Warm-up run (absorbs model-loading overhead) ---------------------------
    print("Warm-up: loading models (may take ~30s on first run)...")
    t_wu_start = time.perf_counter()
    try:
        reset_pc(cpu, LOAD_ADDR)
        cpu.run_gpu_only(max_instructions=MAX_INSTS, batch_size=BATCH_SIZE)
    except (AttributeError, RuntimeError, TypeError) as exc:
        print(f"[ERROR] run_gpu_only() failed: {exc}")
        raise
    t_wu_elapsed = time.perf_counter() - t_wu_start
    print(f"Warm-up done in {t_wu_elapsed:.1f}s")
    print()

    # -- Timed runs -------------------------------------------------------------
    print(f"NeuralCPU run_gpu_only() ({N_RUNS}x):")
    ips_results: list[float] = []
    inst_counts: list[int]   = []
    backend_names: list[str] = []

    for run_idx in range(1, N_RUNS + 1):
        reset_pc(cpu, LOAD_ADDR)
        for _src_addr, dst_addr, payload in expected_copy_windows:
            cpu.memory[dst_addr:dst_addr + len(payload)] = 0
        t_start = time.perf_counter()
        try:
            executed, elapsed = cpu.run_gpu_only(
                max_instructions=MAX_INSTS,
                batch_size=BATCH_SIZE,
            )
        except (AttributeError, RuntimeError, TypeError) as exc:
            print(f"  Run {run_idx}: [ERROR] {exc}")
            continue

        ips = arch_insts / elapsed if elapsed > 0 else 0.0
        ips_results.append(ips)
        inst_counts.append(executed)
        backend = getattr(cpu, "_last_gpu_only_backend", "unknown")
        backend_names.append(str(backend))
        print(
            f"  Run {run_idx}: {executed:>7,} engine-count, "
            f"{elapsed:.4f}s elapsed "
            f"→ {ips:>10,.0f} arch IPS"
            f"  [{backend}]"
        )

    # -- Result verification ----------------------------------------------------
    try:
        x0_val = int(cpu.regs[0].item())
    except Exception:
        x0_val = None

    print()

    # -- Summary ----------------------------------------------------------------
    if ips_results:
        avg_ips = sum(ips_results) / len(ips_results)
        max_ips = max(ips_results)
        min_ips = min(ips_results)
        print(f"Average IPS : {avg_ips:>12,.0f}")
        print(f"Peak IPS    : {max_ips:>12,.0f}")
        print(f"Min IPS     : {min_ips:>12,.0f}")
    else:
        print("No successful runs to report.")
        avg_ips = 0.0
        max_ips = 0.0
        min_ips = 0.0

    print()
    result_line = ""
    result_ok = False
    copy_checks: list[tuple[int, int, bool]] = []
    for _src_addr, dst_addr, payload in expected_copy_windows:
        copied = bytes(cpu.memory[dst_addr:dst_addr + len(payload)].detach().cpu().tolist())
        copy_checks.append((dst_addr, len(payload), copied == payload))
    if x0_val is not None:
        if WORKLOAD == "adjacent-counted":
            ok = "✓" if x0_val == N_ITERS and int(cpu.regs[4].item()) == N_ITERS else "✗"
            status = (
                "OK" if x0_val == N_ITERS and int(cpu.regs[4].item()) == N_ITERS
                else f"MISMATCH (expected X0={N_ITERS}, X4={N_ITERS})"
            )
            result_line = (
                f"Result check: X0 = {x0_val}, X4 = {int(cpu.regs[4].item())} "
                f"(expected {N_ITERS}, {N_ITERS}) {ok}  [{status}]"
            )
            print(result_line)
            result_ok = status == "OK"
        elif WORKLOAD in {"counted-bytecopy", "bytecopy-counted"}:
            counter_val = int(cpu.regs[2].item())
            copies_ok = all(ok for _dst, _size, ok in copy_checks)
            counter_ok = expected_x2 is None or counter_val == expected_x2
            x0_ok = expected_x0 is None or x0_val == expected_x0
            ok = "✓" if x0_ok and counter_ok and copies_ok else "✗"
            status = "OK" if x0_ok and counter_ok and copies_ok else "MISMATCH"
            result_line = (
                f"Result check: X0 = {x0_val} (expected {expected_x0}), "
                f"X2 = {counter_val} (expected {expected_x2}), "
                f"copy windows ok = {sum(ok for _dst, _size, ok in copy_checks)}/{len(copy_checks)} "
                f"{ok}  [{status}]"
            )
            print(result_line)
            result_ok = status == "OK"
        elif WORKLOAD == "bytecopy-cbz-then-bge-exit":
            counter_val = int(cpu.regs[2].item())
            copies_ok = all(ok for _dst, _size, ok in copy_checks)
            counter_ok = counter_val == expected_x2
            ok = "✓" if counter_ok and copies_ok else "✗"
            status = "OK" if counter_ok and copies_ok else "MISMATCH"
            result_line = (
                f"Result check: X2 = {counter_val} (expected {expected_x2}), "
                f"copy windows ok = {sum(ok for _dst, _size, ok in copy_checks)}/{len(copy_checks)} "
                f"{ok}  [{status}]"
            )
            print(result_line)
            result_ok = status == "OK"
        elif WORKLOAD in {"adjacent-bytecopy", "adjacent-bytecopy-bge-exit"}:
            expected_counter = N_ITERS if WORKLOAD == "adjacent-bytecopy-bge-exit" else 0
            counter_val = int(cpu.regs[2].item())
            counter_ok = counter_val == expected_counter
            copies_ok = all(ok for _dst, _size, ok in copy_checks)
            ok = "✓" if counter_ok and copies_ok else "✗"
            status = "OK" if counter_ok and copies_ok else "MISMATCH"
            result_line = (
                f"Result check: X2 = {counter_val} (expected {expected_counter}), "
                f"copy windows ok = {sum(ok for _dst, _size, ok in copy_checks)}/{len(copy_checks)} "
                f"{ok}  [{status}]"
            )
            print(result_line)
            result_ok = status == "OK"
        elif WORKLOAD in {"bytecopy", "bytecopy-cbnz", "bytecopy-cbz-exit", "bytecopy-bge-exit", "bytecopy-blt"}:
            if WORKLOAD in {"bytecopy-blt", "bytecopy-bge-exit"}:
                ok = "✓" if int(cpu.regs[2].item()) == N_ITERS else "✗"
                status = "OK" if int(cpu.regs[2].item()) == N_ITERS else f"MISMATCH (expected X2={N_ITERS})"
                result_line = f"Result check: X2 = {int(cpu.regs[2].item())} (expected {N_ITERS}) {ok}  [{status}]"
            else:
                ok = "✓" if int(cpu.regs[2].item()) == 0 else "✗"
                status = "OK" if int(cpu.regs[2].item()) == 0 else "MISMATCH (expected X2=0)"
                result_line = f"Result check: X2 = {int(cpu.regs[2].item())} (expected 0) {ok}  [{status}]"
            print(result_line)
            result_ok = status == "OK"
        elif WORKLOAD == "nested-counted":
            expected_nested = NESTED_OUTER * NESTED_INNER
            ok = "✓" if int(cpu.regs[2].item()) == expected_nested else "✗"
            status = (
                "OK" if int(cpu.regs[2].item()) == expected_nested
                else f"MISMATCH (expected X2={expected_nested})"
            )
            result_line = (
                f"Result check: X2 = {int(cpu.regs[2].item())} "
                f"(expected {expected_nested}) {ok}  [{status}]"
            )
            print(result_line)
            result_ok = status == "OK"
        else:
            ok = "✓" if x0_val == N_ITERS else "✗"
            status = "OK" if x0_val == N_ITERS else f"MISMATCH (expected {N_ITERS})"
            result_line = f"Result check: X0 = {x0_val} (expected {N_ITERS}) {ok}  [{status}]"
            print(result_line)
            result_ok = status == "OK"
        if inst_counts:
            ok_i = "✓" if inst_counts[-1] == expected_insts else "✗"
            print(
                f"Insts check : {inst_counts[-1]:,} executed "
                f"(expected {expected_insts:,}) {ok_i}"
            )
        if backend_names:
            print(f"Backend    : {backend_names[-1]}")
        print(f"Hotloops   : {int(getattr(cpu, '_last_gpu_only_hotloop_segments', 0))}")
    else:
        print("Result check: could not read X0 register")
    for dst_addr, payload_len, ok_copy in copy_checks:
        print(
            f"Copy check  : 0x{dst_addr:X}..0x{dst_addr + payload_len - 1:X} "
            f"{'✓' if ok_copy else '✗'}  [{'OK' if ok_copy else 'MISMATCH'}]"
        )

    print()

    rust_mod = load_ncpu_metal() if compare_rust else None
    best_rust_backend = None
    best_rust_avg_ips = None
    best_rust_speedup = None
    if rust_mod is not None:
        print("Rust/Metal comparison (same binary, architectural IPS):")
        rust_results = []
        rust_backends = ["FullARM64CPU"]
        if EXPERIMENTAL_RUST:
            rust_backends.extend(["BBCacheMetalCPU", "UltraMetalCPU"])
        rust_setup = None
        rust_verify = None
        if expected_copy_windows:
            def rust_setup(cpu_obj):
                for src_addr, _dst_addr, payload in expected_copy_windows:
                    cpu_obj.write_memory(src_addr, payload)

            def rust_verify(cpu_obj):
                return all(
                    bytes(cpu_obj.read_memory(dst_addr, len(payload))) == payload
                    for _src_addr, dst_addr, payload in expected_copy_windows
                )
        elif WORKLOAD == "nested-counted":
            expected_nested = NESTED_OUTER * NESTED_INNER

            def rust_verify(cpu_obj):
                return cpu_obj.get_register(2) == expected_nested
        elif WORKLOAD == "adjacent-counted":
            def rust_verify(cpu_obj):
                return cpu_obj.get_register(0) == N_ITERS and cpu_obj.get_register(4) == N_ITERS
        elif WORKLOAD == "counted-bytecopy":
            def rust_verify(cpu_obj):
                return (
                    cpu_obj.get_register(0) == N_ITERS and
                    cpu_obj.get_register(2) == 0 and
                    all(
                        bytes(cpu_obj.read_memory(dst_addr, len(payload))) == payload
                        for _src_addr, dst_addr, payload in expected_copy_windows
                    )
                )
        elif WORKLOAD == "bytecopy-counted":
            def rust_verify(cpu_obj):
                return (
                    cpu_obj.get_register(0) == N_ITERS and
                    cpu_obj.get_register(2) == 1 and
                    all(
                        bytes(cpu_obj.read_memory(dst_addr, len(payload))) == payload
                        for _src_addr, dst_addr, payload in expected_copy_windows
                    )
                )
        elif WORKLOAD == "bytecopy-cbz-then-bge-exit":
            def rust_verify(cpu_obj):
                return (
                    cpu_obj.get_register(2) == N_ITERS and
                    all(
                        bytes(cpu_obj.read_memory(dst_addr, len(payload))) == payload
                        for _src_addr, dst_addr, payload in expected_copy_windows
                    )
                )
        for cls_name in rust_backends:
            try:
                result = benchmark_rust_backend(
                    rust_mod, cls_name, code, LOAD_ADDR, arch_insts,
                    setup_fn=rust_setup, verify_fn=rust_verify,
                )
            except Exception as exc:
                result = None
                print(f"  {cls_name:<18} ERROR ({exc})")
            if result is None:
                continue
            status = "OK" if (result["x0_ok"] and result["verify_ok"]) else "BAD RESULT"
            print(
                f"  {cls_name:<18} avg={result['avg_ips']:>10,.0f} "
                f"peak={result['peak_ips']:>10,.0f} "
                f"min={result['min_ips']:>10,.0f} arch IPS  "
                f"[{status}; native cycles={result['native_cycles']:,}]"
            )
            rust_results.append(result)

        if rust_results:
            best = max(rust_results, key=lambda row: row["avg_ips"])
            best_rust_backend = best["name"]
            best_rust_avg_ips = best["avg_ips"]
            print()
            print(
                f"Best Rust/Metal backend on this workload: {best['name']} "
                f"at {best['avg_ips']:,.0f} average arch IPS"
            )
            if ips_results:
                neural_avg = sum(ips_results) / len(ips_results)
                if neural_avg > 0:
                    best_rust_speedup = best["avg_ips"] / neural_avg
                    print(f"Speedup vs NeuralCPU run_gpu_only(): {best_rust_speedup:,.1f}x")
            if not EXPERIMENTAL_RUST:
                print("Set NCPU_BENCH_EXPERIMENTAL_RUST=1 to include BBCacheMetalCPU and UltraMetalCPU.")
            print()

    hotloop_stats = getattr(cpu, "_last_gpu_only_hotloop_stats", {})
    result = {
        "workload": WORKLOAD,
        "avg_ips": avg_ips,
        "peak_ips": max_ips,
        "min_ips": min_ips,
        "backend": backend_names[-1] if backend_names else None,
        "hotloop_segments": int(getattr(cpu, "_last_gpu_only_hotloop_segments", 0)),
        "hotloop_pre_sync_bytes": int(hotloop_stats.get("pre_sync_bytes", 0)),
        "hotloop_post_sync_bytes": int(hotloop_stats.get("flushed_post_sync_bytes", 0)),
        "hotloop_reused_state_segments": int(hotloop_stats.get("reused_state_segments", 0)),
        "hotloop_detector_attempts": int(hotloop_stats.get("detector_attempts", 0)),
        "hotloop_policy_rejections": int(hotloop_stats.get("policy_rejections", 0)),
        "superblock_trace_hits": int(hotloop_stats.get("superblock_trace_hits", 0)),
        "superblock_trace_misses": int(hotloop_stats.get("superblock_trace_misses", 0)),
        "superblock_template_hits": int(hotloop_stats.get("superblock_template_hits", 0)),
        "superblock_template_misses": int(hotloop_stats.get("superblock_template_misses", 0)),
        "superblock_shape_hits": int(hotloop_stats.get("superblock_shape_hits", 0)),
        "result_ok": result_ok,
        "result_check": result_line,
        "executed_count": inst_counts[-1] if inst_counts else None,
        "expected_executed_count": expected_insts,
        "insts_ok": bool(inst_counts and inst_counts[-1] == expected_insts),
        "best_rust_backend": best_rust_backend,
        "best_rust_avg_ips": best_rust_avg_ips,
        "best_rust_speedup_vs_neural": best_rust_speedup,
        "hotloop_samples": list(getattr(cpu, "_last_gpu_only_hotloop_samples", [])),
        "hotloop_trace": list(getattr(cpu, "_last_gpu_only_hotloop_trace", [])),
    }
    backend = result.get("backend")
    if require_backend_prefix is not None:
        backend_ok = isinstance(backend, str) and backend.startswith(require_backend_prefix)
        result["backend_ok"] = bool(backend_ok)
        result["backend_requirement"] = require_backend_prefix
    else:
        result["backend_ok"] = None
        result["backend_requirement"] = None
    WORKLOAD = prev_workload
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=BENCHMARK_WORKLOADS,
        default=None,
        help="Benchmark workload to run (default: environment-selected workload)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the structured benchmark record as JSON",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Write the structured benchmark record to this JSON file",
    )
    parser.add_argument(
        "--compare-rust",
        dest="compare_rust",
        action="store_true",
        default=True,
        help="Include the extra Rust/Metal comparison sweep",
    )
    parser.add_argument(
        "--no-compare-rust",
        dest="compare_rust",
        action="store_false",
        help="Skip the extra Rust/Metal comparison sweep",
    )
    parser.add_argument(
        "--require-backend-prefix",
        default=None,
        help="Mark the result with backend_ok based on the required backend prefix",
    )
    args = parser.parse_args()

    if args.json or args.json_output is not None:
        with redirect_stdout(io.StringIO()):
            record = benchmark(
                args.workload,
                compare_rust=args.compare_rust,
                require_backend_prefix=args.require_backend_prefix,
            )
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(record) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(record))
    else:
        record = benchmark(
            args.workload,
            compare_rust=args.compare_rust,
            require_backend_prefix=args.require_backend_prefix,
        )
        if args.require_backend_prefix and record.get("backend_ok") is False:
            print(
                "Backend requirement not met: "
                f"expected prefix {args.require_backend_prefix!r}, "
                f"observed {record.get('backend')!r}",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = int(main())
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
