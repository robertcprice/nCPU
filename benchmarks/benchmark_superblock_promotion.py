#!/usr/bin/env python3
"""Microbenchmark for adaptive trace-level promotion in the superblock cache.

Construct a workload where the program code is stable but the register file
around the block changes each iteration. Without promotion, every entry spends
a memory-snapshot comparison before falling through to template-level. With
promotion, after N misses we skip the trace-level check entirely.

We measure:
 - trace-level miss count
 - trace-level skip count (promoted entries)
 - template-level hits (should be unchanged; the work just moves earlier)
 - wall-time per lookup

Note: the template-level path re-does cheap comparisons too, so the wall-time
delta is small on tiny workloads. The point of this benchmark is to verify the
skip counter increments as expected and to give a calibration for larger
workloads where the memory snapshot comparison is the expensive step.

Usage:
    python benchmarks/benchmark_superblock_promotion.py
    python benchmarks/benchmark_superblock_promotion.py --iterations 5000
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def _movz(rd: int, imm16: int) -> int:
    return 0xD2800000 | ((imm16 & 0xFFFF) << 5) | rd


def _build_cpu():
    from ncpu.neural.cpu import NeuralCPU
    cpu = NeuralCPU(device_override="cpu", fast_mode=False)
    cpu._hazard_predictor = None
    cpu._dep_predictor = None
    cpu._neural_scheduler = None
    cpu._neural_loop_detector = None
    cpu._neural_branch_predictor = None
    return cpu


def _load(cpu, code: bytes, addr: int) -> None:
    cpu.memory[addr:addr + len(code)] = torch.tensor(
        list(code), dtype=torch.uint8, device=cpu.device
    )


def _reset_pc(cpu, addr: int) -> None:
    cpu.pc = torch.tensor(addr, dtype=torch.int64, device=cpu.device)


def run_one(threshold: int, iterations: int) -> dict:
    os.environ["NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE"] = "4"
    os.environ["NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY"] = "4"
    os.environ["NCPU_GPU_ONLY_SUPERBLOCK_TRACE_PROMOTION"] = str(threshold)

    cpu = _build_cpu()
    cpu._last_gpu_only_hotloop_stats = {}
    cpu._superblock_trace_cache = {}
    cpu._superblock_template_cache = {}
    cpu._superblock_shape_cache = {}
    cpu._superblock_trace_miss_counter = {}

    load_addr = 0x40000
    insts = [_movz(0, 5), _movz(1, 7), 0x00000000]
    _load(cpu, struct.pack(f"<{len(insts)}I", *insts), load_addr)

    # Prime: first entry populates trace + template.
    _reset_pc(cpu, load_addr)
    cpu.regs[5] = 0
    cpu._collect_superblock_candidate()

    # Warm the template bucket by varying a non-guarded register so the trace
    # key differs each time while the template key stays constant.
    t0 = time.perf_counter()
    for i in range(iterations):
        _reset_pc(cpu, load_addr)
        cpu.regs[5] = (i + 1) * 17
        cpu._collect_superblock_candidate()
    wall = time.perf_counter() - t0

    stats = dict(cpu._last_gpu_only_hotloop_stats)
    return {
        "threshold": threshold,
        "iterations": iterations,
        "wall_seconds": round(wall, 4),
        "wall_per_iter_us": round(wall / max(iterations, 1) * 1e6, 2),
        "cache_misses": int(stats.get("superblock_cache_misses", 0)),
        "cache_hits": int(stats.get("superblock_cache_hits", 0)),
        "template_hits": int(stats.get("superblock_template_hits", 0)),
        "template_misses": int(stats.get("superblock_template_misses", 0)),
        "trace_skips": int(stats.get("superblock_trace_skips", 0)),
        "trace_promotions": int(stats.get("superblock_trace_promotions", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[0, 3, 8],
        help="trace promotion thresholds to compare (0 disables)",
    )
    args = parser.parse_args()

    print(f"Iterations per configuration: {args.iterations}")
    print()
    print(f"{'threshold':>10} {'wall_s':>10} {'us/iter':>10} "
          f"{'trace_miss':>10} {'trace_skip':>10} {'trace_promo':>12} "
          f"{'tpl_hit':>8}")
    for threshold in args.thresholds:
        row = run_one(threshold, args.iterations)
        print(
            f"{row['threshold']:>10} "
            f"{row['wall_seconds']:>10.4f} "
            f"{row['wall_per_iter_us']:>10.2f} "
            f"{row['cache_misses']:>10} "
            f"{row['trace_skips']:>10} "
            f"{row['trace_promotions']:>12} "
            f"{row['template_hits']:>8}"
        )
    print()
    print("Interpretation:")
    print("  threshold=0 disables adaptive promotion (always check trace level)")
    print("  threshold>0 skips trace-level lookup after N consecutive trace misses")
    print("  trace_skips indicates how many lookups were promoted past trace-level")
    print("  template_hits should be unchanged across thresholds — only the work")
    print("  that precedes it differs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
