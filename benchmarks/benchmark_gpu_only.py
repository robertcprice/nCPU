#!/usr/bin/env python3
"""run_gpu_only() IPS Benchmark.

Measures the throughput of the zero-sync GPU-only execution engine on a tight
ARM64 arithmetic loop (N=2000 iterations).  The program computes X0 = N by
adding 1 to X0 on every iteration, giving a known expected result for
correctness verification.

Workload layout (7 instructions, ~2003 total executed):
  MOVZ X0, #0          ; acc = 0
  MOVZ X1, #N          ; counter = N
  MOVZ X2, #1          ; step = 1
  ADD  X0, X0, X2      ; loop: acc += step
  SUBS X1, X1, #1      ;       counter--  (sets flags)
  B.NE loop            ;       repeat if counter != 0
  HALT                 ; 0x00000000

Expected result: X0 = N (2000), X1 = 0.

Usage:
    python benchmarks/benchmark_gpu_only.py

Expected output (MPS, Apple Silicon M-series):
    run_gpu_only() IPS: ~30,000 – 36,000   (fully zero-sync GPU execution)
"""

import struct
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────

N_ITERS      = 2000          # loop iteration count
LOAD_ADDR    = 0x10000       # same load address used by other benchmarks
N_RUNS       = 3             # timed measurement repetitions
MAX_INSTS    = 200_000       # upper bound; program halts well before this
BATCH_SIZE   = 64            # matches run_gpu_only() default

# ── ARM64 program builder ─────────────────────────────────────────────────────

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


# ── Main benchmark ────────────────────────────────────────────────────────────

def benchmark() -> None:
    print()
    print("nCPU run_gpu_only() IPS Benchmark")
    print("=" * 50)
    print(f"Program: tight ADD+SUBS+B.NE loop, N={N_ITERS} iterations")
    print(f"Expected result: X0 = {N_ITERS} (each iteration adds step=1)")
    print()

    # -- Import -----------------------------------------------------------------
    try:
        import torch
        from ncpu.neural.cpu import NeuralCPU
    except ImportError as exc:
        print(f"[ERROR] Could not import NeuralCPU: {exc}")
        print("        Make sure you are running from the project root.")
        sys.exit(1)

    # -- Build program ----------------------------------------------------------
    code = build_program(N_ITERS)
    n_program_insts = len(code) // 4
    # total instructions executed: 3 (init MOVZ) + N*(ADD+SUBS+BNE) + 1 (HALT)
    expected_insts = 3 + N_ITERS * 3 + 1
    print(f"Program size: {n_program_insts} instructions ({len(code)} bytes)")
    print(f"Expected instructions to execute: {expected_insts:,}")
    print()

    # -- Create CPU (fast_mode=False uses neural ALU) ---------------------------
    try:
        cpu = NeuralCPU(fast_mode=False)
    except Exception as exc:
        print(f"[ERROR] NeuralCPU construction failed: {exc}")
        sys.exit(1)

    # -- Load program once ------------------------------------------------------
    load_program(cpu, code, LOAD_ADDR)

    # -- Warm-up run (absorbs model-loading overhead) ---------------------------
    print("Warm-up: loading models (may take ~30s on first run)...")
    t_wu_start = time.perf_counter()
    try:
        reset_pc(cpu, LOAD_ADDR)
        cpu.run_gpu_only(max_instructions=MAX_INSTS, batch_size=BATCH_SIZE)
    except (AttributeError, RuntimeError, TypeError) as exc:
        print(f"[ERROR] run_gpu_only() failed: {exc}")
        sys.exit(1)
    t_wu_elapsed = time.perf_counter() - t_wu_start
    print(f"Warm-up done in {t_wu_elapsed:.1f}s")
    print()

    # -- Timed runs -------------------------------------------------------------
    print(f"Benchmark runs ({N_RUNS}x):")
    ips_results: list[float] = []
    inst_counts: list[int]   = []

    for run_idx in range(1, N_RUNS + 1):
        reset_pc(cpu, LOAD_ADDR)
        t_start = time.perf_counter()
        try:
            executed, elapsed = cpu.run_gpu_only(
                max_instructions=MAX_INSTS,
                batch_size=BATCH_SIZE,
            )
        except (AttributeError, RuntimeError, TypeError) as exc:
            print(f"  Run {run_idx}: [ERROR] {exc}")
            continue

        ips = executed / elapsed if elapsed > 0 else 0.0
        ips_results.append(ips)
        inst_counts.append(executed)
        print(
            f"  Run {run_idx}: {executed:>7,} insts, "
            f"{elapsed:.4f}s elapsed "
            f"→ {ips:>10,.0f} IPS"
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

    print()
    if x0_val is not None:
        ok = "✓" if x0_val == N_ITERS else "✗"
        status = "OK" if x0_val == N_ITERS else f"MISMATCH (expected {N_ITERS})"
        print(f"Result check: X0 = {x0_val} (expected {N_ITERS}) {ok}  [{status}]")
        if inst_counts:
            ok_i = "✓" if inst_counts[-1] == expected_insts else "✗"
            print(
                f"Insts check : {inst_counts[-1]:,} executed "
                f"(expected {expected_insts:,}) {ok_i}"
            )
    else:
        print("Result check: could not read X0 register")

    print()


if __name__ == "__main__":
    benchmark()
