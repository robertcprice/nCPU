#!/usr/bin/env python3
"""
Benchmark: Neural Terminal Renderer throughput.

Measures rendering speed across diverse terminal content, both single-frame
and batched modes. Reports FPS, ms/frame, and pixel throughput.

Usage:
    python benchmarks/benchmark_neural_display.py
    python benchmarks/benchmark_neural_display.py --device cpu
    python benchmarks/benchmark_neural_display.py --iters 200
"""

import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ncpu.neural.neural_terminal_renderer import (
    NeuralDisplay, NeuralTerminalRenderer, TerminalState,
    TERM_ROWS, TERM_COLS, FRAME_H, FRAME_W,
)

MODEL_PATH = Path(__file__).parent.parent / 'models' / 'display' / 'terminal_renderer.pt'


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()


def benchmark_single_frame(display, device, n_iters=100):
    """Benchmark single-frame rendering with diverse content."""
    scenes = [
        ("plain text", "Hello World\n" * 10),
        ("colored text", "\x1b[31mRed\x1b[32mGreen\x1b[34mBlue\x1b[0m " * 20),
        ("all ASCII", ''.join(chr(c) for c in range(32, 127)) * 3),
        ("code", "def fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)\n" * 4),
    ]

    results = []
    for name, text in scenes:
        display.reset()
        display.terminal.write_str(text)

        # Warm up
        for _ in range(5):
            display.render()
        sync(device)

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(n_iters):
            frame = display.render()
        sync(device)
        elapsed = time.perf_counter() - t0

        ms = elapsed / n_iters * 1000
        fps = n_iters / elapsed
        results.append((name, ms, fps))

    return results


def benchmark_batch(renderer, device, batch_sizes=[1, 2, 4, 8, 16], n_iters=50):
    """Benchmark batched rendering at various batch sizes."""
    renderer.eval()
    results = []

    for bs in batch_sizes:
        chars = torch.randint(32, 127, (bs, TERM_ROWS, TERM_COLS), device=device)
        fg = torch.randint(0, 16, (bs, TERM_ROWS, TERM_COLS), device=device)
        bg = torch.zeros(bs, TERM_ROWS, TERM_COLS, dtype=torch.long, device=device)

        # Warm up
        with torch.no_grad():
            for _ in range(5):
                renderer(chars, fg, bg)
        sync(device)

        # Benchmark
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                out = renderer(chars, fg, bg)
        sync(device)
        elapsed = time.perf_counter() - t0

        total_frames = n_iters * bs
        ms = elapsed / total_frames * 1000
        fps = total_frames / elapsed
        results.append((bs, ms, fps))

    return results


def benchmark_components(renderer, device, n_iters=200):
    """Benchmark individual pipeline components."""
    renderer.eval()
    chars = torch.randint(32, 127, (TERM_ROWS, TERM_COLS), device=device)
    fg = torch.randint(0, 16, (TERM_ROWS, TERM_COLS), device=device)
    bg = torch.zeros(TERM_ROWS, TERM_COLS, dtype=torch.long, device=device)

    results = []

    # Glyph generation
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            alpha = renderer.glyphs(chars)
    sync(device)
    ms = (time.perf_counter() - t0) / n_iters * 1000
    results.append(("glyph_gen", ms))

    # Color lookup
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            fg_rgb = renderer.colors(fg)
            bg_rgb = renderer.colors(bg)
    sync(device)
    ms = (time.perf_counter() - t0) / n_iters * 1000
    results.append(("color_lookup", ms))

    # Compositor
    frame = torch.rand(1, 3, FRAME_H, FRAME_W, device=device)
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            out = renderer.compositor(frame)
    sync(device)
    ms = (time.perf_counter() - t0) / n_iters * 1000
    results.append(("compositor", ms))

    return results


def main():
    parser = argparse.ArgumentParser(description="Neural Display Benchmark")
    parser.add_argument('--device', default=None)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    print("=" * 60)
    print("  Neural Terminal Renderer — Rendering Benchmark")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Frame size: {FRAME_W}x{FRAME_H} ({FRAME_W*FRAME_H*3:,} bytes)")
    print(f"  Model:      {MODEL_PATH.name}")
    print(f"  Iterations: {args.iters}")

    if not MODEL_PATH.exists():
        print(f"\nModel not found: {MODEL_PATH}")
        sys.exit(1)

    display = NeuralDisplay(str(MODEL_PATH), device=device)
    renderer = display.renderer
    n_params = renderer.count_params()
    print(f"  Parameters: {n_params:,}")
    print()

    # Single frame benchmark
    print("--- Single Frame Rendering ---")
    print(f"{'Scene':<16} {'ms/frame':>10} {'FPS':>8} {'MB/s':>8}")
    print("-" * 46)
    single_results = benchmark_single_frame(display, device, args.iters)
    for name, ms, fps in single_results:
        throughput = (FRAME_W * FRAME_H * 3 * fps) / (1024 * 1024)
        print(f"{name:<16} {ms:>9.1f}  {fps:>7.0f}  {throughput:>7.0f}")

    avg_ms = np.mean([r[1] for r in single_results])
    avg_fps = np.mean([r[2] for r in single_results])
    print("-" * 46)
    print(f"{'AVERAGE':<16} {avg_ms:>9.1f}  {avg_fps:>7.0f}  "
          f"{(FRAME_W*FRAME_H*3*avg_fps)/(1024*1024):>7.0f}")
    print()

    # Batch benchmark
    print("--- Batched Rendering ---")
    print(f"{'Batch':>6} {'ms/frame':>10} {'FPS':>8} {'Speedup':>8}")
    print("-" * 36)
    batch_results = benchmark_batch(renderer, device, n_iters=max(20, args.iters // 5))
    base_fps = batch_results[0][2] if batch_results else 1
    for bs, ms, fps in batch_results:
        speedup = fps / base_fps
        print(f"{bs:>6} {ms:>9.1f}  {fps:>7.0f}  {speedup:>7.1f}x")
    print()

    # Component benchmark
    print("--- Component Breakdown ---")
    print(f"{'Component':<16} {'ms':>8} {'%':>6}")
    print("-" * 32)
    comp_results = benchmark_components(renderer, device, args.iters)
    total_comp = sum(r[1] for r in comp_results)
    for name, ms in comp_results:
        pct = ms / total_comp * 100
        print(f"{name:<16} {ms:>7.2f}  {pct:>5.1f}%")
    print("-" * 32)
    print(f"{'total':<16} {total_comp:>7.2f}  100.0%")
    overhead = avg_ms - total_comp
    if overhead > 0:
        print(f"{'overhead':<16} {overhead:>7.2f}  (numpy/tensor conversion)")
    print()

    # Summary
    print("=" * 60)
    print(f"  Neural Display: {avg_fps:.0f} FPS single, "
          f"{batch_results[-1][2]:.0f} FPS batch-{batch_results[-1][0]}")
    print(f"  Latency: {avg_ms:.1f}ms single, {batch_results[-1][1]:.1f}ms batched")
    pixel_rate = FRAME_W * FRAME_H * avg_fps
    print(f"  Pixel rate: {pixel_rate/1e6:.1f}M pixels/s")
    print(f"  All {n_params:,} parameters are neural — zero conventional rasterization")
    print("=" * 60)


if __name__ == '__main__':
    main()
