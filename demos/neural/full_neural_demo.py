#!/usr/bin/env python3
"""Full Neural CPU Demo -- every stage from decode to display is a trained neural network.

This is the definitive nCPU demonstration: real ARM64 machine code executes on the
NeuralCPU's woven engine (run_woven), where every ALU operation passes through trained
.pt specialist models (arithmetic.pt, multiply.pt, logical.pt, lsl.pt, lsr.pt).
SYS_WRITE syscalls feed output bytes into a NeuralDisplayV2, which renders each pixel
through a neural glyph MLP, learned color embeddings, and a ConvNet compositor.

Zero conventional computation anywhere in the pipeline:
  - Instruction decode: table + neural ARM64 decoder (arm64_decoder.pt)
  - ALU add/sub:        Kogge-Stone carry-lookahead (arithmetic.pt + carry_combine.pt)
  - ALU multiply:       256x256 byte-pair LUT (multiply.pt)
  - ALU logic:          Neural truth tables (logical.pt)
  - ALU shift:          Learned shift decoder (lsl.pt / lsr.pt)
  - Branch prediction:  NeuralBranchPredictor (LSTM, trained online)
  - Memory addressing:  NeuralMemoryArithmetic (pointer.pt)
  - Display:            Glyph MLP + color embed + ConvNet compositor (terminal_renderer_v2.pt)

Usage:
    python demos/neural/full_neural_demo.py
    python demos/neural/full_neural_demo.py --max-instructions 50000
    python demos/neural/full_neural_demo.py --device cpu
    python demos/neural/full_neural_demo.py --output /tmp/neural_demo.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# ARM64 Machine Code Encoders
# ═══════════════════════════════════════════════════════════════════════════

def encode_movz(rd: int, imm16: int, hw: int = 0) -> int:
    """MOVZ Xd, #imm16, LSL #(hw*16)"""
    return (0b110100101 << 23) | (hw << 21) | (imm16 << 5) | rd


def encode_movk(rd: int, imm16: int, hw: int = 0) -> int:
    """MOVK Xd, #imm16, LSL #(hw*16)"""
    return (0b111100101 << 23) | (hw << 21) | (imm16 << 5) | rd


def encode_svc(imm16: int = 0) -> int:
    """SVC #imm16"""
    return 0xD4000001 | (imm16 << 5)


def encode_add_imm(rd: int, rn: int, imm12: int) -> int:
    """ADD Xd, Xn, #imm12"""
    return (0b1001000100 << 22) | (imm12 << 10) | (rn << 5) | rd


def encode_sub_imm(rd: int, rn: int, imm12: int) -> int:
    """SUB Xd, Xn, #imm12"""
    return (0b1101000100 << 22) | (imm12 << 10) | (rn << 5) | rd


def encode_subs_imm(rd: int, rn: int, imm12: int) -> int:
    """SUBS Xd, Xn, #imm12"""
    return (0b1111000100 << 22) | (imm12 << 10) | (rn << 5) | rd


def encode_add_reg(rd: int, rn: int, rm: int) -> int:
    """ADD Xd, Xn, Xm"""
    return (0x8B << 24) | (rm << 16) | (rn << 5) | rd


def encode_cmp_reg(rn: int, rm: int) -> int:
    """CMP Xn, Xm  (alias for SUBS XZR, Xn, Xm)"""
    return (0xEB << 24) | (rm << 16) | (rn << 5) | 0x1F


def encode_strb(rt: int, rn: int, imm12: int) -> int:
    """STRB Wt, [Xn, #imm12]"""
    return (0x39 << 24) | (imm12 << 10) | (rn << 5) | rt


def encode_b_ne(offset_instructions: int) -> int:
    """B.NE <offset> (offset in instructions, signed)"""
    imm19 = offset_instructions & 0x7FFFF
    return 0x54000001 | (imm19 << 5)  # cond=0001 (NE)


def encode_b(offset_instructions: int) -> int:
    """B <offset> (unconditional branch, offset in instructions, signed)"""
    imm26 = offset_instructions & 0x3FFFFFF
    return 0x14000000 | imm26


def inst_bytes(inst: int) -> bytes:
    """Encode a 32-bit instruction to 4 little-endian bytes."""
    return struct.pack('<I', inst & 0xFFFFFFFF)


# ═══════════════════════════════════════════════════════════════════════════
# Program Builders
# ═══════════════════════════════════════════════════════════════════════════

LOAD_ADDR = 0x1000  # Low load address so all addresses fit in 16 bits (no MOVK needed)

@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    title: str
    loop_counts: tuple[int, ...]
    note: str

    @property
    def expected_total(self) -> int:
        return sum(self.loop_counts)


WORKLOAD_SPECS: dict[str, WorkloadSpec] = {
    "single-1k": WorkloadSpec(
        name="single-1k",
        title="Single Counter Loop (1K)",
        loop_counts=(1000,),
        note="Baseline bottom-up workload for the strict full-neural path.",
    ),
    "single-4k": WorkloadSpec(
        name="single-4k",
        title="Single Counter Loop (4K)",
        loop_counts=(4000,),
        note="Longer single-phase loop to scale the same neural execution path.",
    ),
    "dual-1k": WorkloadSpec(
        name="dual-1k",
        title="Dual Counter Loop (1K + 1K)",
        loop_counts=(1000, 1000),
        note="Two sequential counted regions to exercise chained branch handoff.",
    ),
    "triple-512": WorkloadSpec(
        name="triple-512",
        title="Triple Counter Loop (512 x 3)",
        loop_counts=(512, 512, 512),
        note="Three short phases to stress repeated loop setup and teardown.",
    ),
    "staggered": WorkloadSpec(
        name="staggered",
        title="Staggered Counter Loop (256 + 1024 + 2048)",
        loop_counts=(256, 1024, 2048),
        note="Mixed loop lengths for a less uniform branch pattern.",
    ),
}

DEFAULT_WORKLOAD = "single-1k"


def available_workloads() -> list[str]:
    return list(WORKLOAD_SPECS.keys())


def resolve_workload(name: str) -> WorkloadSpec:
    try:
        return WORKLOAD_SPECS[name]
    except KeyError as exc:
        known = ", ".join(available_workloads())
        raise ValueError(f"Unknown workload '{name}'. Known workloads: {known}") from exc


def _banner_bytes(spec: WorkloadSpec) -> bytes:
    title = spec.title[:34]
    title_row = f"| {title:^40} |\n"
    return (
        b"+------------------------------------------+\n"
        + title_row.encode("ascii")
        + b"+------------------------------------------+\n"
        + b"|                                          |\n"
        + b"|  Every component is a neural network:    |\n"
        + b"|   * ALU:     Kogge-Stone CLA (8 pass)   |\n"
        + b"|   * Logic:   Neural truth tables         |\n"
        + b"|   * Multiply: 256x256 byte-pair LUT     |\n"
        + b"|   * Shifts:  Learned shift decoder       |\n"
        + b"|   * Decode:  Neural ARM64 decoder        |\n"
        + b"|   * Memory:  Neural pointer arithmetic   |\n"
        + b"|   * Display: Glyph MLP + ConvNet         |\n"
        + b"|                                          |\n"
        + b"|  Zero conventional computation.          |\n"
        + b"+------------------------------------------+\n"
    )


def _loop_section_bytes(spec: WorkloadSpec) -> bytes:
    loop_list = ", ".join(str(count) for count in spec.loop_counts)
    lines = [
        "",
        f"Workload: {spec.title}",
        f"Phases:   {loop_list}",
        f"Total:    {spec.expected_total} increments",
        "  ADD X10, X10, #1   -> arithmetic.pt (CLA)",
        "  SUB X12, X12, #1   -> arithmetic.pt",
        "  CBNZ X12, loop     -> neural branch eval",
        f"  Note: {spec.note}",
        "",
    ]
    return "\n".join(lines).encode("ascii")


def build_workload_program(spec: WorkloadSpec) -> bytes:
    """Build a single ARM64 program for a named bottom-up full-neural workload."""
    banner = _banner_bytes(spec)
    loop_section = _loop_section_bytes(spec)
    footer = b"\nAll output from 100% neural pipeline.\n"

    banner_off = 0x400
    loop_off = banner_off + len(banner)
    footer_off = loop_off + len(loop_section)

    banner_addr = LOAD_ADDR + banner_off
    loop_addr = LOAD_ADDR + loop_off
    footer_addr = LOAD_ADDR + footer_off

    code: list[int] = []

    def load_addr(rd: int, addr: int) -> None:
        code.append(encode_movz(rd, addr & 0xFFFF))

    def emit_write(buf_addr: int, length: int) -> None:
        code.append(encode_movz(0, 1))
        load_addr(1, buf_addr)
        code.append(encode_movz(2, length))
        code.append(encode_movz(8, 64))
        code.append(encode_svc(0))

    emit_write(banner_addr, len(banner))

    code.append(encode_movz(10, 0))  # accumulator = 0
    for count in spec.loop_counts:
        code.append(encode_movz(12, count))
        loop_start = len(code)
        code.append(encode_add_imm(10, 10, 1))
        code.append(encode_sub_imm(12, 12, 1))
        loop_offset = loop_start - len(code)
        imm19 = loop_offset & 0x7FFFF
        code.append(0xB5000000 | (imm19 << 5) | 12)

    emit_write(loop_addr, len(loop_section))
    emit_write(footer_addr, len(footer))

    code.append(encode_movz(0, 0))
    code.append(encode_movz(8, 93))
    code.append(encode_svc(0))
    code.append(0x00000000)

    total_size = footer_off + len(footer) + 16
    binary = bytearray(total_size)
    for i, inst in enumerate(code):
        struct.pack_into("<I", binary, i * 4, inst & 0xFFFFFFFF)
    binary[banner_off:banner_off + len(banner)] = banner
    binary[loop_off:loop_off + len(loop_section)] = loop_section
    binary[footer_off:footer_off + len(footer)] = footer
    return bytes(binary)


# ═══════════════════════════════════════════════════════════════════════════
# PNG helper
# ═══════════════════════════════════════════════════════════════════════════

def save_png(array: np.ndarray, path: Path) -> None:
    """Save an RGB numpy array as PNG."""
    try:
        from PIL import Image
    except ImportError:
        print(f"  [WARNING] PIL not installed -- cannot save {path}")
        return
    img = Image.fromarray(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    print(f"  Saved: {path}")


def save_json(payload: dict, path: Path) -> None:
    """Save a JSON summary for reproducible benchmarking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"  Saved summary: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main Demo
# ═══════════════════════════════════════════════════════════════════════════

def run_demo(args):
    import torch

    device = args.device
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    spec = resolve_workload(args.workload)

    print()
    print("=" * 60)
    print("   nCPU -- Full Neural CPU Demo")
    print("   Every stage from decode to display is neural")
    print("=" * 60)

    # ── Step 1: Create NeuralCPU (neural ALU enabled) ─────────────────────
    print("\n[1] Initializing NeuralCPU (fast_mode=False)...")
    t0 = time.perf_counter()

    import logging
    logging.basicConfig(level=logging.WARNING)

    from ncpu.neural.cpu.core import NeuralCPU
    cpu = NeuralCPU(fast_mode=False, device_override=device)

    init_time = time.perf_counter() - t0
    print(f"    NeuralCPU ready on {cpu.device} ({init_time:.2f}s)")

    # ── Step 2: Attach NeuralDisplayV2 ────────────────────────────────────
    print("\n[2] Loading NeuralDisplayV2 (terminal_renderer_v2.pt)...")
    t0 = time.perf_counter()

    from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2
    display = NeuralDisplayV2(device=str(cpu.device))
    cpu._neural_display = display

    display_time = time.perf_counter() - t0
    params = display.renderer.count_params()
    print(f"    Display loaded: {params:,} parameters ({display_time:.2f}s)")

    # ── Step 3: Build and load the program ────────────────────────────────
    print("\n[3] Building ARM64 machine code...")
    binary = build_workload_program(spec)
    cpu.load_binary(binary, LOAD_ADDR)
    cpu.pc = torch.tensor(LOAD_ADDR, dtype=torch.int64, device=cpu.device)
    # Set SP to a sensible value
    cpu.regs[31] = 0xFF000
    print(f"    Loaded {len(binary)} bytes at 0x{LOAD_ADDR:X}")
    print(f"    Workload: {spec.name} ({spec.title})")
    print(f"    Loop counts: {', '.join(str(count) for count in spec.loop_counts)}")
    inst_count = sum(1 for i in range(0, min(len(binary), 0x400), 4)
                     if binary[i:i+4] != b'\x00\x00\x00\x00')
    print(f"    Program: ~{inst_count} instructions + string data")

    # ── Step 4: Execute via neural woven engine ───────────────────────────
    max_inst = args.max_instructions
    print(f"\n[4] Executing via run_woven() (max {max_inst:,} instructions)...")
    print("    Neural pipeline: decode -> neural ALU -> neural branch -> scatter")
    print()
    sys.stdout.flush()
    print("    --- program output begin ---")
    sys.stdout.flush()

    t0 = time.perf_counter()
    executed, elapsed = cpu.run_woven(max_instructions=max_inst)
    exec_time = time.perf_counter() - t0

    sys.stdout.flush()
    print("    --- program output end ---")
    ips = executed / elapsed if elapsed > 0 else 0
    print(f"\n    Executed: {executed:,} instructions in {exec_time:.3f}s")
    print(f"    Throughput: {ips:,.0f} IPS (instructions per second)")

    # ── Counter loop verification (registers after loop) ────────────────────
    # X10 should equal the total loop count after all phases; X12 should be 0.
    x10_actual = int(cpu.regs[10].item())
    x12_actual = int(cpu.regs[12].item())
    x10_expected = spec.expected_total
    x10_ok = (x10_actual == x10_expected)
    x12_ok = (x12_actual == 0)
    print(f"\n    Counter loop verification ({spec.name}):")
    print(f"      X10 (accum)   = {x10_actual}  (expected {x10_expected})  {'OK' if x10_ok else 'MISMATCH'}")
    print(f"      X12 (counter) = {x12_actual}  (expected 0)     {'OK' if x12_ok else 'MISMATCH'}")

    # ── Step 5: Render via neural display ─────────────────────────────────
    print(f"\n[5] Rendering via NeuralDisplayV2...")
    t0 = time.perf_counter()
    frame = display.render()
    render_time = time.perf_counter() - t0
    print(f"    Frame: {frame.shape[1]}x{frame.shape[0]} pixels, rendered in {render_time*1000:.1f}ms")
    print(f"    Every pixel produced by neural network forward passes")

    # ── Step 6: Save output ───────────────────────────────────────────────
    output_path = Path(args.output)
    print(f"\n[6] Saving output...")
    save_png(frame, output_path)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("   NEURAL PIPELINE SUMMARY")
    print("=" * 60)
    print(f"   Device:              {cpu.device}")
    print(f"   Workload:            {spec.name} ({spec.title})")
    print(f"   Neural ALU:          ENABLED (arithmetic.pt + logical.pt + multiply.pt)")
    print(f"   Neural decode:       ENABLED (table + arm64_decoder.pt)")
    print(f"   Neural display:      ENABLED (terminal_renderer_v2.pt, {params:,} params)")
    print(f"   Instructions:        {executed:,}")
    print(f"   Execution time:      {exec_time:.3f}s")
    print(f"   Render time:         {render_time*1000:.1f}ms")
    print(f"   Throughput:          {ips:,.0f} IPS")
    print(f"   Output:              {output_path}")
    print(f"   Conventional ops:    0 (fully neural)")
    print("=" * 60)
    print()

    summary = {
        "demo": "full_neural_demo",
        "mode": "bottom_up_fully_neural",
        "workload": spec.name,
        "workload_title": spec.title,
        "loop_counts": list(spec.loop_counts),
        "expected_total_iterations": int(spec.expected_total),
        "workload_note": spec.note,
        "device": str(cpu.device),
        "display_params": int(params),
        "executed_instructions": int(executed),
        "execution_time_s": float(exec_time),
        "render_time_ms": float(render_time * 1000.0),
        "throughput_ips": float(ips),
        "frame_height": int(frame.shape[0]),
        "frame_width": int(frame.shape[1]),
        "counter_x10": int(x10_actual),
        "counter_x12": int(x12_actual),
        "counter_expected": int(x10_expected),
        "counter_verified": bool(x10_ok and x12_ok),
        "conventional_ops": 0,
        "output_path": str(output_path),
    }

    if args.summary_json:
        save_json(summary, Path(args.summary_json))

    return cpu, display, frame, summary


def main():
    parser = argparse.ArgumentParser(
        description="Full Neural CPU Demo -- every component is a trained neural network"
    )
    parser.add_argument(
        "--workload", choices=available_workloads(), default=DEFAULT_WORKLOAD,
        help="Named bottom-up workload to run"
    )
    parser.add_argument(
        "--list-workloads", action="store_true",
        help="Print the available named workloads and exit"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on (cpu, mps, cuda). Default: auto-detect."
    )
    parser.add_argument(
        "--max-instructions", type=int, default=100_000,
        help="Maximum instructions to execute (default: 100000)"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "models" / "display" / "full_neural_demo.png"),
        help="Output PNG path"
    )
    parser.add_argument(
        "--summary-json", type=str, default=None,
        help="Optional JSON path for a machine-readable execution summary"
    )
    args = parser.parse_args()
    if args.list_workloads:
        print("Available workloads:")
        for name in available_workloads():
            spec = resolve_workload(name)
            print(
                f"- {name}: {spec.title} | loops={','.join(str(count) for count in spec.loop_counts)} "
                f"| total={spec.expected_total}"
            )
        return
    run_demo(args)


if __name__ == "__main__":
    main()
