#!/usr/bin/env python3
"""Comprehensive benchmark of all nCPU execution paths.

Runs the same ADD-loop program across every available execution engine
and reports IPS for each.

Usage:
    python benchmarks/benchmark_neural_computer.py
"""

import sys
import time
import struct
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── ARM64 encoding helpers ────────────────────────────────────────────────

def movz_w(rd, imm16):
    return (0b01010010100 << 21) | (imm16 << 5) | rd

def add_imm_w(rd, rn, imm12):
    return (0x11 << 24) | (imm12 << 10) | (rn << 5) | rd

def subs_reg_w(rd, rn, rm):
    return (0x6B << 24) | (rm << 16) | (rn << 5) | rd

def b_ne(offset):
    imm19 = offset & 0x7FFFF
    return 0x54000001 | (imm19 << 5)

def movz_x(rd, imm16):
    return (0b110100101 << 23) | (imm16 << 5) | rd

def svc(imm=0):
    return 0xD4000001 | (imm << 5)

def build_loop_program(n_iterations):
    """Build: counter=0, limit=N, loop: ADD counter,#1; SUBS cmp; B.NE loop; EXIT"""
    code = [
        movz_w(0, 0),                    # W0 = 0 (counter)
        movz_w(1, n_iterations & 0xFFFF),# W1 = N
        add_imm_w(0, 0, 1),             # loop: ADD W0, W0, #1
        subs_reg_w(2, 0, 1),            # SUBS W2, W0, W1
        b_ne(-2),                         # B.NE loop
        movz_x(0, 0),                    # X0 = 0 (exit code)
        movz_x(8, 93),                   # X8 = SYS_EXIT
        svc(0),                           # SVC #0
    ]
    return b''.join(struct.pack('<I', i) for i in code)


def benchmark_metal_conventional(binary, n_iters):
    """Benchmark FullARM64CPU (conventional Metal, ~1M IPS)."""
    try:
        import importlib.util
        so_path = PROJECT_ROOT / 'kernels' / 'rust_metal' / 'ncpu_metal.abi3.so'
        if not so_path.exists():
            return None, "ncpu_metal.abi3.so not found"
        spec = importlib.util.spec_from_file_location('ncpu_metal', str(so_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if not hasattr(mod, 'FullARM64CPU'):
            return None, "FullARM64CPU not found"

        cpu = mod.FullARM64CPU()
        cpu.load_program(binary, 0x10000)
        cpu.set_pc(0x10000)

        t0 = time.perf_counter()
        result = cpu.execute(max_cycles=n_iters * 5 + 200)
        dt = time.perf_counter() - t0

        ips = result.total_cycles / dt if dt > 0 else 0
        return ips, f"{result.total_cycles} cycles in {dt*1000:.1f}ms"
    except Exception as e:
        return None, str(e)


def benchmark_metal_neural(binary, n_iters):
    """Benchmark NeuralFullARM64CPU (neural Metal, cooperative threadgroup)."""
    try:
        import importlib.util
        so_path = PROJECT_ROOT / 'kernels' / 'rust_metal' / 'ncpu_metal.abi3.so'
        if not so_path.exists():
            return None, "ncpu_metal.abi3.so not found"
        spec = importlib.util.spec_from_file_location('ncpu_metal', str(so_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if not hasattr(mod, 'NeuralFullARM64CPU'):
            return None, "NeuralFullARM64CPU not found"

        from tests.test_neural_cpu import load_neural_weights, load_mul_lut
        cc, tt = load_neural_weights()
        mul = load_mul_lut()
        dummy_shift = [0.0] * (64 * 64 * 64)

        cpu = mod.NeuralFullARM64CPU(memory_size=4 * 1024 * 1024)
        cpu.load_neural_weights(cc, tt)
        cpu.load_mul_lut(mul)
        cpu.load_shift_luts(dummy_shift, dummy_shift)

        cpu.load_program(binary, 0x10000)
        cpu.set_pc(0x10000)

        t0 = time.perf_counter()
        result = cpu.execute(max_cycles=n_iters * 5 + 200)
        dt = time.perf_counter() - t0

        ips = result.total_cycles / dt if dt > 0 else 0
        return ips, f"{result.total_cycles} cycles in {dt*1000:.1f}ms"
    except Exception as e:
        return None, str(e)


def benchmark_pytorch_woven(n_iters):
    """Benchmark NeuralCPU.run_woven() (PyTorch neural ALU)."""
    try:
        from ncpu.neural.cpu.core import NeuralCPU
        import torch

        cpu = NeuralCPU(fast_mode=False, device_override='cpu')
        binary = build_loop_program(n_iters)

        # Load into NeuralCPU memory
        load_addr = 0x10000
        for i, b in enumerate(binary):
            cpu.memory[load_addr + i] = b
        cpu.pc = torch.tensor(load_addr, dtype=torch.int64, device=cpu.device)
        cpu.regs[31] = 0xFF000  # SP

        t0 = time.perf_counter()
        executed, _ = cpu.run_woven(max_instructions=n_iters * 5 + 200)
        dt = time.perf_counter() - t0

        ips = executed / dt if dt > 0 else 0
        return ips, f"{executed} instructions in {dt*1000:.1f}ms"
    except Exception as e:
        return None, str(e)


def benchmark_display_fps():
    """Benchmark NeuralDisplayV2 render speed."""
    try:
        from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2
        display = NeuralDisplayV2(device='mps')
        display.write(b"Hello from neural display benchmark!\n" * 20)

        # Warm up
        display.render()

        # Time 10 renders
        t0 = time.perf_counter()
        n = 10
        for _ in range(n):
            display.render()
        dt = time.perf_counter() - t0

        fps = n / dt
        return fps, f"{n} frames in {dt*1000:.0f}ms"
    except Exception as e:
        return None, str(e)


def main():
    N = 1000  # loop iterations

    print()
    print("=" * 70)
    print("  nCPU Neural Computer — Comprehensive Benchmark")
    print(f"  Test program: {N}-iteration ADD loop")
    print("=" * 70)
    print()

    binary = build_loop_program(N)

    results = []

    # 1. Metal conventional
    print("  [1/4] Metal GPU (conventional)...", end=" ", flush=True)
    ips, note = benchmark_metal_conventional(binary, N)
    if ips:
        print(f"{ips:,.0f} IPS  ({note})")
        results.append(("Metal GPU (conventional)", ips, "No", "FullARM64CPU"))
    else:
        print(f"SKIP ({note})")

    # 2. Metal neural (cooperative threadgroup)
    print("  [2/4] Metal Neural CPU (cooperative)...", end=" ", flush=True)
    ips, note = benchmark_metal_neural(binary, N)
    if ips:
        print(f"{ips:,.0f} IPS  ({note})")
        results.append(("Metal Neural CPU (coop)", ips, "All ALU", "Kogge-Stone CLA"))
    else:
        print(f"SKIP ({note})")

    # 3. PyTorch woven
    print("  [3/4] PyTorch run_woven()...", end=" ", flush=True)
    ips, note = benchmark_pytorch_woven(N)
    if ips:
        print(f"{ips:,.0f} IPS  ({note})")
        results.append(("PyTorch run_woven()", ips, "All ALU", "Neural weave"))
    else:
        print(f"SKIP ({note})")

    # 4. Neural display
    print("  [4/4] Neural Display V2...", end=" ", flush=True)
    fps, note = benchmark_display_fps()
    if fps:
        print(f"{fps:.0f} FPS  ({note})")
        results.append(("Neural Display V2", fps, "Display", "Glyph MLP"))
    else:
        print(f"SKIP ({note})")

    # Summary table
    print()
    print("=" * 70)
    print(f"  {'Execution Path':<30} {'Speed':>12} {'Neural?':<10} {'Engine'}")
    print("  " + "-" * 66)
    for name, speed, neural, engine in results:
        unit = "FPS" if "Display" in name else "IPS"
        print(f"  {name:<30} {speed:>10,.0f} {unit}  {neural:<10} {engine}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
