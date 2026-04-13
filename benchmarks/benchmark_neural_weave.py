#!/usr/bin/env python3
"""Neural Weave IPS Benchmark.

Compares three execution modes on a tight 500-iteration ARM64 arithmetic loop:
  1. neural-serial  — step() with use_neural_alu=True     (~2K IPS, fully neural, serial)
  2. fast-parallel  — run_parallel_gpu(), tensor ops only  (varies — tight loops may slow)
  3. woven          — run_woven(), neural ALU + vectorizer (~20K IPS, fully neural)

The woven mode uses a neural loop vectorizer: for tight backward B.NE loops it
detects the accumulator pattern, computes N*delta via neural MUL, and applies the
accumulated result in one neural ADD — processing 500 iterations in ~3 neural calls
instead of 1500 per-instruction batches.

Usage:
    python benchmarks/benchmark_neural_weave.py

Expected output (MPS, Apple Silicon M-series):
    [neural-serial ] IPS: ~2,000    fully neural ✓
    [fast-parallel ] IPS: varies
    [woven         ] IPS: ~20,000   fully neural ✓ (9× faster than serial neural)
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ncpu.neural.cpu import NeuralCPU


# ── Minimal ARM64 workload: tight arithmetic loop ─────────────────────────────
# Computes sum(0..N-1) using ADD + CMP + B.NE
# Exercises: MOVZ, ADD_IMM, ADD_REG, SUB_IMM, CMP_REG, B_COND

LOOP_COUNT = 500  # iterations — enough to warm up batch paths

def build_loop_program(n_iters: int) -> bytes:
    """Assemble a tight ARM64 loop into raw bytes.

    x0 = accumulator (0 → sum)
    x1 = counter    (n_iters → 0)
    x2 = step       (1)

    loop:
        add x0, x0, x2     // acc += step
        subs x1, x1, #1    // counter--; set flags
        b.ne loop           // if counter != 0, repeat
        // fall through → halt (0x00000000)
    """
    insts: list[int] = []

    # MOVZ x0, #0        (acc = 0)
    # Encoding: sf=1, opc=10, hw=00, imm16=0, Rd=0
    insts.append(0xD2800000)

    # MOVZ x1, #n_iters  (counter)
    insts.append(0xD2800001 | ((n_iters & 0xFFFF) << 5))

    # MOVZ x2, #1        (step = 1)
    insts.append(0xD2800022)

    # Loop body (3 instructions starting at byte offset 12):
    loop_offset = len(insts) * 4

    # ADD x0, x0, x2     (64-bit ADD reg)
    # Encoding: ADD Xd, Xn, Xm = 0x8B000000 | Rm<<16 | Rn<<5 | Rd
    insts.append(0x8B020000)  # ADD x0, x0, x2

    # SUBS x1, x1, #1    (subtract immediate, sets flags)
    # Encoding: 0xF1000000 | imm12<<10 | Rn<<5 | Rd
    insts.append(0xF1000421)  # SUBS x1, x1, #1

    # B.NE loop           (branch if Z==0, offset = -(2 instructions) = -8 bytes)
    # Encoding: 0x54000001 | imm19<<5 | cond=1 (NE)
    branch_offset = loop_offset - (len(insts) * 4)
    imm19 = (branch_offset >> 2) & 0x7FFFF
    insts.append(0x54000001 | (imm19 << 5))

    # HALT (0x00000000)
    insts.append(0x00000000)

    import struct
    return struct.pack(f'<{len(insts)}I', *insts)


def load_program(cpu: NeuralCPU, code: bytes, load_addr: int = 0x10000):
    """Write raw ARM64 bytes into the CPU's memory tensor."""
    for i, b in enumerate(code):
        cpu.memory[load_addr + i] = b
    cpu.pc = torch.tensor(load_addr, dtype=torch.int64, device=cpu.device)
    # Zero all registers
    cpu.regs[:] = 0
    cpu.flags[:] = 0
    cpu.halted = False


def run_serial_neural(cpu: NeuralCPU, code: bytes, n_insts: int) -> float:
    """Run n_insts instructions in serial neural mode. Returns elapsed seconds."""
    load_program(cpu, code)
    start = time.perf_counter()
    for _ in range(n_insts):
        if cpu.halted:
            break
        cpu.step()
    return time.perf_counter() - start


def run_parallel_fast(cpu: NeuralCPU, code: bytes) -> tuple[int, float]:
    """Run in fast parallel mode (tensor ops, no neural ALU)."""
    load_program(cpu, code)
    return cpu.run_parallel_gpu(max_instructions=2_000_000, batch_size=256)


def run_woven(cpu: NeuralCPU, code: bytes) -> tuple[int, float]:
    """Run in neural weave mode (fully neural ALU, batched)."""
    load_program(cpu, code)
    return cpu.run_woven(max_instructions=2_000_000, batch_size=256)


# ── Benchmark runner ──────────────────────────────────────────────────────────

def benchmark():
    print("=" * 64)
    print("  Neural Weave IPS Benchmark")
    print("=" * 64)

    code = build_loop_program(LOOP_COUNT)
    insts_in_loop = 3 * LOOP_COUNT + 3  # MOVZ×3 + (ADD+SUBS+BNE)×N + halt

    print(f"  Workload: {LOOP_COUNT}-iteration arithmetic loop")
    print(f"  Expected instructions: ~{insts_in_loop:,}")
    print()

    # ── 1. Neural serial (step-by-step, true neural ALU) ─────────────────────
    print("  [1/3] neural-serial  (step(), use_neural_alu=True) …")
    cpu_n = NeuralCPU(fast_mode=False)
    # Warmup
    load_program(cpu_n, code)
    cpu_n.step()
    # Timed (limit to 200 steps to avoid multi-minute wait)
    n_serial = min(200, insts_in_loop)
    elapsed_s = run_serial_neural(cpu_n, code, n_serial)
    ips_serial = n_serial / elapsed_s if elapsed_s > 0 else 0
    print(f"      {n_serial} insts in {elapsed_s:.3f}s → {ips_serial:,.0f} IPS  (neural ✓)")

    # ── 2. Fast parallel (tensor ops, no neural ALU) ──────────────────────────
    print("  [2/3] fast-parallel  (run_parallel_gpu(), tensor ops) …")
    cpu_f = NeuralCPU(fast_mode=True)
    # Warmup
    load_program(cpu_f, code)
    cpu_f.run_parallel_gpu(max_instructions=100, batch_size=32)
    # Timed
    executed_f, elapsed_f = run_parallel_fast(cpu_f, code)
    ips_fast = executed_f / elapsed_f if elapsed_f > 0 else 0
    print(f"      {executed_f:,} insts in {elapsed_f:.4f}s → {ips_fast:,.0f} IPS  (neural ✗, tensor ops)")

    # ── 3. Woven (neural ALU, batched) ───────────────────────────────────────
    print("  [3/3] woven          (run_woven(), neural ALU) …")
    cpu_w = NeuralCPU(fast_mode=False)
    # Warmup (first batch has model-load overhead)
    load_program(cpu_w, code)
    cpu_w.run_woven(max_instructions=50, batch_size=32)
    # Timed
    executed_w, elapsed_w = run_woven(cpu_w, code)
    ips_woven = executed_w / elapsed_w if elapsed_w > 0 else 0
    print(f"      {executed_w:,} insts in {elapsed_w:.4f}s → {ips_woven:,.0f} IPS  (neural ✓)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print(f"  {'Mode':<18}  {'IPS':>12}  {'vs serial':>12}  Neural?")
    print(f"  {'-'*18}  {'-'*12}  {'-'*12}  -------")
    print(f"  {'neural-serial':<18}  {ips_serial:>12,.0f}  {'1×':>12}  ✓")
    speedup_f = ips_fast / ips_serial if ips_serial > 0 else 0
    speedup_w = ips_woven / ips_serial if ips_serial > 0 else 0
    print(f"  {'fast-parallel':<18}  {ips_fast:>12,.0f}  {speedup_f:>11.0f}×  ✗ (tensor ops)")
    print(f"  {'woven':<18}  {ips_woven:>12,.0f}  {speedup_w:>11.0f}×  ✓ (neural ALU)")
    print("=" * 64)

    if ips_woven > ips_serial:
        ratio = ips_woven / ips_serial
        print(f"\n  Neural Weave is {ratio:.0f}× faster than serial neural mode")
        print(f"  and {ips_woven / ips_fast * 100:.1f}% of pure tensor-op speed.")
    else:
        print("\n  NOTE: woven mode appears slower than expected — check model loading.")


if __name__ == "__main__":
    benchmark()
