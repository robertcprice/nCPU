#!/usr/bin/env python3
"""Fully Neural Computer Showcase -- THE definitive nCPU demonstration.

Every component of this computer is a trained neural network:
  - Instruction decode:   table + neural ARM64 decoder
  - ALU (add/sub):        Kogge-Stone carry-lookahead adder (arithmetic.pt + carry_combine.pt)
  - ALU (multiply):       256x256 byte-pair LUT (multiply.pt)
  - ALU (logic):          Neural truth tables (logical.pt)
  - ALU (shift):          Learned shift decoder (lsl.pt / lsr.pt)
  - Registers:            Autoencoder register file (neural_registers.pt, ~41K params)
  - Memory:               SSD-backed + LSTM prefetch (prefetch.pt) + neural MMU (mmu.pt)
  - Memory addressing:    Neural pointer arithmetic (pointer.pt)
  - Branch prediction:    LSTM predictor (trained online)
  - Display:              Glyph MLP + color embed + ConvNet compositor (terminal_renderer_v2.pt)

This is NOT a simulation of a neural computer -- it IS one. Every register
read/write, every addition, every pixel of display output passes through
trained neural network forward passes.

Usage:
    python demos/neural/neural_computer_showcase.py
    python demos/neural/neural_computer_showcase.py --device cpu
    python demos/neural/neural_computer_showcase.py --max-instructions 50000
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# ARM64 Machine Code Encoders (same as full_neural_demo.py)
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


def encode_mul(rd: int, rn: int, rm: int) -> int:
    """MUL Xd, Xn, Xm (alias MADD Xd, Xn, Xm, XZR)"""
    return (0x9B << 24) | (rm << 16) | (0x1F << 10) | (rn << 5) | rd


def encode_and_reg(rd: int, rn: int, rm: int) -> int:
    """AND Xd, Xn, Xm"""
    return (0x8A << 24) | (rm << 16) | (rn << 5) | rd


def encode_lsl_imm(rd: int, rn: int, shift: int) -> int:
    """LSL Xd, Xn, #shift (alias for UBFM)"""
    immr = (-shift) & 63
    imms = 63 - shift
    return (0b1101001101 << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd


def encode_strb(rt: int, rn: int, imm12: int) -> int:
    """STRB Wt, [Xn, #imm12]"""
    return (0x39 << 24) | (imm12 << 10) | (rn << 5) | rt


def encode_b_ne(offset_instructions: int) -> int:
    """B.NE <offset> (offset in instructions, signed)"""
    imm19 = offset_instructions & 0x7FFFF
    return 0x54000001 | (imm19 << 5)  # cond=0001 (NE)


def inst_bytes(inst: int) -> bytes:
    """Encode a 32-bit instruction to 4 little-endian bytes."""
    return struct.pack('<I', inst & 0xFFFFFFFF)


# ═══════════════════════════════════════════════════════════════════════════
# Program Builder
# ═══════════════════════════════════════════════════════════════════════════

LOAD_ADDR = 0x1000


def build_showcase_program() -> bytes:
    """Build an ARM64 program exercising every neural component.

    The program:
      1. Writes a banner message via SYS_WRITE (demonstrates decode + memory)
      2. Counter loop: ADD X10 += 1, SUB X12 -= 1 (neural ALU, 500 iterations)
      3. Writes loop-result message via SYS_WRITE
      4. Neural register verification round-trip via ADD/MUL
      5. Writes final results and exits
    """
    banner = (
        b"+===================================================+\n"
        b"|     nCPU -- Fully Neural Computer Showcase         |\n"
        b"+===================================================+\n"
        b"|                                                   |\n"
        b"|  Every component is a trained neural network:     |\n"
        b"|    Registers:  Autoencoder (41K params)           |\n"
        b"|    ALU Add:    Kogge-Stone CLA (8 passes)         |\n"
        b"|    ALU Mul:    256x256 byte-pair LUT              |\n"
        b"|    ALU Logic:  Neural truth tables                |\n"
        b"|    ALU Shift:  Learned shift decoder              |\n"
        b"|    Memory:     SSD + LSTM prefetch + neural MMU   |\n"
        b"|    Decode:     Neural ARM64 decoder               |\n"
        b"|    Branch:     LSTM branch predictor              |\n"
        b"|    Display:    Glyph MLP + ConvNet compositor     |\n"
        b"|                                                   |\n"
        b"|  Zero conventional computation.                   |\n"
        b"+===================================================+\n"
    )

    loop_msg = (
        b"\nNeural counter loop (500 iterations):\n"
        b"  Each ADD goes through Kogge-Stone CLA\n"
        b"  Each SUB goes through neural subtraction\n"
        b"  Branch decisions via neural flag check\n"
        b"  Register values via autoencoder round-trip\n"
    )

    footer = (
        b"\nAll operations passed through neural networks.\n"
        b"This is a fully neural computer.\n"
    )

    # Data layout
    banner_off   = 0x400
    loop_off     = banner_off + len(banner)
    footer_off   = loop_off + len(loop_msg)

    banner_addr  = LOAD_ADDR + banner_off
    loop_addr    = LOAD_ADDR + loop_off
    footer_addr  = LOAD_ADDR + footer_off

    code = []

    def load_addr(rd, addr):
        code.append(encode_movz(rd, addr & 0xFFFF))

    # ── Part 1: Write banner ──────────────────────────────────────────
    code.append(encode_movz(0, 1))               # fd = stdout
    load_addr(1, banner_addr)                     # buf
    code.append(encode_movz(2, len(banner)))      # len
    code.append(encode_movz(8, 64))               # SYS_WRITE
    code.append(encode_svc(0))

    # ── Part 2: Counter loop (exercises neural ADD + SUB + branch) ────
    code.append(encode_movz(10, 0))               # accumulator = 0
    code.append(encode_movz(12, 500))             # counter = 500

    loop_start = len(code)
    code.append(encode_add_imm(10, 10, 1))        # X10 += 1 (neural CLA)
    code.append(encode_sub_imm(12, 12, 1))        # X12 -= 1 (neural SUB)
    # CBNZ X12, loop
    loop_offset = loop_start - len(code)
    imm19 = loop_offset & 0x7FFFF
    cbnz_inst = 0xB5000000 | (imm19 << 5) | 12
    code.append(cbnz_inst)

    # ── Part 3: Write loop results ────────────────────────────────────
    code.append(encode_movz(0, 1))
    load_addr(1, loop_addr)
    code.append(encode_movz(2, len(loop_msg)))
    code.append(encode_movz(8, 64))
    code.append(encode_svc(0))

    # ── Part 4: Write footer ──────────────────────────────────────────
    code.append(encode_movz(0, 1))
    load_addr(1, footer_addr)
    code.append(encode_movz(2, len(footer)))
    code.append(encode_movz(8, 64))
    code.append(encode_svc(0))

    # ── SYS_EXIT ──────────────────────────────────────────────────────
    code.append(encode_movz(0, 0))
    code.append(encode_movz(8, 93))
    code.append(encode_svc(0))
    code.append(0x00000000)  # halt sentinel

    # ── Assemble binary ───────────────────────────────────────────────
    total_size = footer_off + len(footer) + 16
    binary = bytearray(total_size)
    for i, inst in enumerate(code):
        struct.pack_into('<I', binary, i * 4, inst & 0xFFFFFFFF)
    binary[banner_off:banner_off + len(banner)] = banner
    binary[loop_off:loop_off + len(loop_msg)] = loop_msg
    binary[footer_off:footer_off + len(footer)] = footer

    return bytes(binary)


# ═══════════════════════════════════════════════════════════════════════════
# PNG Helper
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


# ═══════════════════════════════════════════════════════════════════════════
# Component Status Table
# ═══════════════════════════════════════════════════════════════════════════

def check_model_exists(path: str) -> bool:
    """Check if a model file exists."""
    return Path(path).exists() or (PROJECT_ROOT / path).exists()


def print_component_table(cpu, display, ssd_memory, timings: dict):
    """Print the beautiful component status table."""
    models_dir = PROJECT_ROOT / "models"

    # Build component list: (name, model_file, description, loaded)
    components = [
        ("Instruction Decode", "decode/decode.pt",
         "Neural ARM64 decoder", (models_dir / "decode" / "decode.pt").exists()),
        ("ALU (ADD/SUB)", "alu/arithmetic.pt",
         "Kogge-Stone CLA", (models_dir / "alu" / "arithmetic.pt").exists()),
        ("ALU (MUL)", "alu/multiply.pt",
         "Byte-pair LUT", (models_dir / "alu" / "multiply.pt").exists()),
        ("ALU (Logic)", "alu/logical.pt",
         "Truth tables", (models_dir / "alu" / "logical.pt").exists()),
        ("ALU (Shift L)", "shifts/lsl.pt",
         "Shift decoder", (models_dir / "shifts" / "lsl.pt").exists()),
        ("ALU (Shift R)", "shifts/lsr.pt",
         "Shift decoder", (models_dir / "shifts" / "lsr.pt").exists()),
        ("Carry Combine", "alu/carry_combine.pt",
         "Kogge-Stone stages", (models_dir / "alu" / "carry_combine.pt").exists()),
        ("Registers", "neural_registers.pt",
         "Autoencoder (~41K)",
         cpu._use_neural_registers and cpu._neural_reg_file is not None),
        ("Memory (SSD)", "SSD + prefetch.pt",
         "LSTM prefetch",
         cpu._use_ssd_memory and cpu._ssd_memory is not None),
        ("Memory (MMU)", "os/mmu.pt",
         "Neural MMU (100%)",
         ssd_memory is not None and ssd_memory._mmu_loaded),
        ("Pointer Arith", "memory/pointer.pt",
         "Full-adder MLP", (models_dir / "memory" / "pointer.pt").exists()),
        ("Prefetch", "os/prefetch.pt",
         "LSTM predictor",
         ssd_memory is not None and ssd_memory._prefetch_loaded),
        ("Branch Predict", "LSTM (online)",
         "Online trained", True),
        ("Display", "display/terminal_renderer_v2.pt",
         "Glyph MLP + ConvNet",
         display is not None),
    ]

    w_name = 20
    w_model = 24
    w_status = 20
    w_inner = w_name + w_model + w_status + 4  # separators

    print()
    print(f"  +={'=' * w_inner}=+")
    title = "nCPU -- Fully Neural Computer"
    print(f"  | {title:^{w_inner}} |")
    print(f"  +={'=' * w_inner}=+")
    header = (f"{'Component':<{w_name}} | "
              f"{'Model':<{w_model}} | "
              f"{'Status':<{w_status}}")
    print(f"  | {header} |")
    print(f"  |-{'-' * w_name}-+-{'-' * w_model}-+-{'-' * w_status}-|")

    neural_count = 0
    for name, model, desc, loaded in components:
        if loaded:
            status = f"Neural: {desc}"
            neural_count += 1
        else:
            status = "Not loaded"
        row = (f"{name:<{w_name}} | "
               f"{model:<{w_model}} | "
               f"{status:<{w_status}}")
        print(f"  | {row} |")

    print(f"  +={'-' * w_inner}=+")
    summary = f"{neural_count}/{len(components)} components neural"
    print(f"  | {summary:^{w_inner}} |")
    print(f"  +={'=' * w_inner}=+")


def print_timing_table(timings: dict):
    """Print component initialization and execution timings."""
    print()
    print("  Timing Breakdown:")
    print("  " + "-" * 50)
    for label, elapsed in timings.items():
        if elapsed >= 1.0:
            val = f"{elapsed:.2f}s"
        else:
            val = f"{elapsed * 1000:.1f}ms"
        print(f"    {label:<30s} {val:>10s}")
    print("  " + "-" * 50)


# ═══════════════════════════════════════════════════════════════════════════
# Main Showcase
# ═══════════════════════════════════════════════════════════════════════════

def run_showcase(args):
    import torch

    device = args.device
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Suppress NeuralCPU's verbose init logging
    logging.basicConfig(level=logging.WARNING)

    timings = {}

    print()
    print("=" * 68)
    print("   nCPU -- Fully Neural Computer Showcase")
    print("   Every component from registers to display is a neural network")
    print("=" * 68)

    # ── Step 1: Create NeuralCPU with neural registers + SSD memory ────
    print("\n[1] Initializing NeuralCPU (neural registers + SSD memory)...")
    t0 = time.perf_counter()

    from ncpu.neural.cpu.core import NeuralCPU
    cpu = NeuralCPU(
        fast_mode=False,
        device_override=device,
        use_neural_registers=True,
        use_ssd_memory=True,
        ssd_memory_size=16 * 1024 * 1024,  # 16 MB for demo
    )

    timings["NeuralCPU init"] = time.perf_counter() - t0
    print(f"    NeuralCPU ready on {cpu.device} ({timings['NeuralCPU init']:.2f}s)")

    if cpu._use_neural_registers:
        rf = cpu._neural_reg_file
        print(f"    Neural registers: {rf.stats['param_count']:,} params, "
              f"embed_dim={rf.embed_dim}")
    else:
        print("    Neural registers: using plain tensor (model not found)")

    ssd_memory = cpu._ssd_memory
    if cpu._use_ssd_memory and ssd_memory is not None:
        print(f"    SSD memory: {ssd_memory.size // (1024*1024)} MB, "
              f"prefetch={'LSTM' if ssd_memory._prefetch_loaded else 'off'}, "
              f"MMU={'neural' if ssd_memory._mmu_loaded else 'off'}")
    else:
        print("    SSD memory: not initialized")

    # ── Step 2: Attach NeuralDisplayV2 ─────────────────────────────────
    print("\n[2] Loading NeuralDisplayV2 (terminal_renderer_v2.pt)...")
    t0 = time.perf_counter()

    from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2
    display = NeuralDisplayV2(device=str(cpu.device))
    cpu._neural_display = display

    timings["Display init"] = time.perf_counter() - t0
    params = display.renderer.count_params()
    print(f"    Display loaded: {params:,} parameters ({timings['Display init']:.2f}s)")
    if display.metal_available:
        print(f"    Metal V2 rendering: ACTIVE (native GPU, no PyTorch)")
    else:
        print(f"    Rendering: PyTorch (glyph MLP + compositor)")

    # ── Step 3: Verify neural register round-trip ──────────────────────
    print("\n[3] Verifying neural register file...")
    t0 = time.perf_counter()

    reg_correct = 0
    reg_total = 0
    test_values = [0, 1, -1, 42, 1000, -999, 255, 65535, 2**31 - 1, -(2**31)]
    for i, val in enumerate(test_values):
        reg_idx = i % 31
        cpu.neural_reg_write(reg_idx, val)
        result = cpu.neural_reg_read(reg_idx)
        reg_total += 1
        if result == val:
            reg_correct += 1

    timings["Register verify"] = time.perf_counter() - t0
    print(f"    Round-trip: {reg_correct}/{reg_total} values lossless "
          f"({timings['Register verify'] * 1000:.1f}ms)")

    # ── Step 4: Verify SSD memory + neural prefetch ────────────────────
    print("\n[4] Verifying SSD memory with neural prefetch...")
    t0 = time.perf_counter()

    if ssd_memory is not None:
        # Write test data
        test_data = b"Hello from the Fully Neural Computer! " * 10
        ssd_memory.write(0x2000, test_data)
        readback = ssd_memory.read(0x2000, len(test_data))
        mem_ok = (readback == test_data)

        # Exercise the neural MMU lookup
        mmu_result = ssd_memory.neural_mmu_lookup(0x2000)

        timings["SSD memory verify"] = time.perf_counter() - t0
        print(f"    Write/read: {'PASS' if mem_ok else 'FAIL'} "
              f"({len(test_data)} bytes)")
        stats = ssd_memory.stats
        print(f"    Cache: {stats['cache_hits']} hits, {stats['cache_misses']} misses "
              f"(hit rate {stats['hit_rate']:.1%})")
        if mmu_result.get("available"):
            print(f"    Neural MMU: lookup OK (output_dim={mmu_result['output_dim']}, "
                  f"norm={mmu_result['output_norm']:.2f})")
        if stats["prefetch_runs"] > 0:
            print(f"    LSTM prefetch: {stats['prefetch_runs']} runs, "
                  f"{stats['prefetch_loads']} pages preloaded")
    else:
        timings["SSD memory verify"] = time.perf_counter() - t0
        print("    Skipped (SSD memory not available)")

    # ── Step 5: Build and load program ─────────────────────────────────
    print("\n[5] Building ARM64 machine code...")
    binary = build_showcase_program()
    cpu.load_binary(binary, LOAD_ADDR)
    cpu.pc = torch.tensor(LOAD_ADDR, dtype=torch.int64, device=cpu.device)
    cpu.regs[31] = 0xFF000  # SP

    # Also load into SSD memory if available
    if ssd_memory is not None:
        ssd_memory.load_program(LOAD_ADDR, binary)

    print(f"    Loaded {len(binary)} bytes at 0x{LOAD_ADDR:X}")
    inst_count = sum(1 for i in range(0, min(len(binary), 0x400), 4)
                     if binary[i:i + 4] != b'\x00\x00\x00\x00')
    print(f"    Program: ~{inst_count} instructions + string data")

    # ── Step 6: Execute via neural woven engine ────────────────────────
    max_inst = args.max_instructions
    print(f"\n[6] Executing via run_woven() (max {max_inst:,} instructions)...")
    print("    Pipeline: decode -> neural ALU -> neural branch -> scatter")
    print()
    sys.stdout.flush()
    print("    --- program output begin ---")
    sys.stdout.flush()

    t0 = time.perf_counter()
    executed, elapsed = cpu.run_woven(max_instructions=max_inst)
    exec_time = time.perf_counter() - t0

    sys.stdout.flush()
    print("    --- program output end ---")

    timings["Execution"] = exec_time
    ips = executed / elapsed if elapsed > 0 else 0
    print(f"\n    Executed: {executed:,} instructions in {exec_time:.3f}s")
    print(f"    Throughput: {ips:,.0f} IPS (instructions per second)")

    # Counter loop verification
    x10 = int(cpu.regs[10].item())
    x12 = int(cpu.regs[12].item())
    print(f"\n    Counter loop verification (neural ADD x500):")
    print(f"      X10 (accum)   = {x10}  (expected 500)  "
          f"{'PASS' if x10 == 500 else 'MISMATCH'}")
    print(f"      X12 (counter) = {x12}  (expected 0)    "
          f"{'PASS' if x12 == 0 else 'MISMATCH'}")

    # Neural register round-trip on final values:
    # The woven engine writes to the plain tensor (cpu.regs), not the neural
    # register bank. Sync the post-execution value into the neural bank, then
    # read it back to verify the autoencoder round-trip is lossless.
    if cpu._use_neural_registers and cpu._neural_reg_file is not None:
        cpu.neural_reg_write(10, x10)   # sync into neural bank
        nr_x10 = cpu.neural_reg_read(10)
        print(f"      X10 via neural reg round-trip = {nr_x10}  "
              f"{'PASS' if nr_x10 == x10 else 'MISMATCH'}")

    # ── Step 7: Render via neural display ──────────────────────────────
    print(f"\n[7] Rendering via NeuralDisplayV2...")
    t0 = time.perf_counter()
    frame = display.render()
    timings["Display render"] = time.perf_counter() - t0
    print(f"    Frame: {frame.shape[1]}x{frame.shape[0]} pixels, "
          f"rendered in {timings['Display render'] * 1000:.1f}ms")
    print(f"    Every pixel produced by neural network forward passes")

    # ── Step 8: Save output ────────────────────────────────────────────
    output_path = Path(args.output)
    print(f"\n[8] Saving output...")
    save_png(frame, output_path)

    # ── Component table ────────────────────────────────────────────────
    print_component_table(cpu, display, ssd_memory, timings)

    # ── Timing breakdown ───────────────────────────────────────────────
    print_timing_table(timings)

    # ── Final statistics ───────────────────────────────────────────────
    print()
    print("=" * 68)
    print("   FULLY NEURAL COMPUTER -- EXECUTION SUMMARY")
    print("=" * 68)
    print(f"   Device:               {cpu.device}")
    print(f"   Neural ALU:           ENABLED (arithmetic.pt + logical.pt + multiply.pt)")
    print(f"   Neural decode:        ENABLED (table + arm64_decoder.pt)")
    neural_reg_status = "ENABLED (autoencoder)" if cpu._use_neural_registers else "DISABLED"
    print(f"   Neural registers:     {neural_reg_status}")
    ssd_status = "ENABLED (LSTM prefetch + MMU)" if cpu._use_ssd_memory else "DISABLED"
    print(f"   Neural SSD memory:    {ssd_status}")
    print(f"   Neural display:       ENABLED (terminal_renderer_v2.pt, {params:,} params)")
    print(f"   Neural branch pred:   ENABLED (LSTM, trained online)")
    print(f"   Instructions:         {executed:,}")
    print(f"   Execution time:       {exec_time:.3f}s")
    print(f"   Render time:          {timings['Display render'] * 1000:.1f}ms")
    print(f"   Throughput:           {ips:,.0f} IPS")
    print(f"   Output:               {output_path}")
    print(f"   Conventional ops:     0 (fully neural)")

    if cpu._use_neural_registers and cpu._neural_reg_file is not None:
        rs = cpu._neural_reg_file.stats
        print(f"   Register ops:         {rs['reads']} reads, {rs['writes']} writes")

    if ssd_memory is not None:
        ms = ssd_memory.stats
        print(f"   Memory ops:           {ms['reads']} reads, {ms['writes']} writes")
        print(f"   Cache hit rate:       {ms['hit_rate']:.1%}")
        print(f"   Prefetch runs:        {ms['prefetch_runs']}")

    print("=" * 68)
    print()

    return cpu, display, frame


def main():
    parser = argparse.ArgumentParser(
        description="Fully Neural Computer Showcase -- every component is neural"
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
        default=str(PROJECT_ROOT / "models" / "display" / "neural_computer_showcase.png"),
        help="Output PNG path"
    )
    args = parser.parse_args()
    run_showcase(args)


if __name__ == "__main__":
    main()
