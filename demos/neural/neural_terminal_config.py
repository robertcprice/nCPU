#!/usr/bin/env python3
"""Neural Terminal Configuration Utility — inspect, benchmark, and configure.

A CLI tool (not a visual demo) for inspecting the neural terminal rendering
system: model info, benchmark speeds across backends, palette management,
and weight export for the Metal shader.

Usage:
    python demos/neural/neural_terminal_config.py --info
    python demos/neural/neural_terminal_config.py --benchmark
    python demos/neural/neural_terminal_config.py --benchmark --frames 200
    python demos/neural/neural_terminal_config.py --palette
    python demos/neural/neural_terminal_config.py --export-weights /tmp/weights.npy
    python demos/neural/neural_terminal_config.py --info --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from ncpu.neural.neural_terminal_renderer import (
    NeuralDisplay,
    NeuralTerminalRenderer,
    NeuralGlyphGenerator,
    NeuralColorPalette,
    NeuralCompositor,
    TerminalState,
    ANSI_PALETTE,
    FRAME_H, FRAME_W,
    TERM_ROWS, TERM_COLS,
    CELL_H, CELL_W,
    N_CHARS, N_COLORS,
)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    """Print a section header."""
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _row(label: str, value: str, width: int = 30) -> None:
    """Print a key-value row."""
    print(f"  {label:<{width}} {value}")


def _table(headers: list[str], rows: list[list[str]], col_widths: list[int]) -> None:
    """Print a formatted table."""
    # Header
    hdr = "  "
    sep = "  "
    for h, w in zip(headers, col_widths):
        hdr += f"{h:<{w}} "
        sep += "-" * w + " "
    print(hdr)
    print(sep)
    for row in rows:
        line = "  "
        for cell, w in zip(row, col_widths):
            line += f"{cell:<{w}} "
        print(line)


# ---------------------------------------------------------------------------
# --info: Model information
# ---------------------------------------------------------------------------

def cmd_info() -> None:
    """Display model information and system capabilities."""
    _header("Neural Terminal Renderer - Model Information")

    models_dir = PROJECT_ROOT / "models" / "display"

    # V1 model
    v1_path = models_dir / "terminal_renderer.pt"
    v1_exists = v1_path.exists()
    v1_size = v1_path.stat().st_size if v1_exists else 0

    # V2 model
    v2_path = models_dir / "terminal_renderer_v2.pt"
    v2_exists = v2_path.exists()
    v2_size = v2_path.stat().st_size if v2_exists else 0

    print()
    _row("Terminal geometry:", f"{TERM_ROWS} rows x {TERM_COLS} cols")
    _row("Cell size:", f"{CELL_H}x{CELL_W} pixels")
    _row("Frame size:", f"{FRAME_W}x{FRAME_H} pixels ({FRAME_W * FRAME_H * 3:,} bytes/frame)")
    _row("Character set (V1):", f"{N_CHARS} characters")
    _row("Color palette (V1):", f"{N_COLORS} colors (ANSI 16)")

    # V1 architecture
    _header("V1 Renderer Architecture")
    v1_renderer = NeuralTerminalRenderer()
    total = v1_renderer.count_params()
    glyph_params = sum(p.numel() for p in v1_renderer.glyphs.parameters())
    color_params = sum(p.numel() for p in v1_renderer.colors.parameters())
    comp_params = sum(p.numel() for p in v1_renderer.compositor.parameters())

    print()
    _table(
        ["Component", "Parameters", "Description"],
        [
            ["NeuralGlyphGenerator", f"{glyph_params:,}", f"char embed({N_CHARS},64) + MLP(64->256->256->128)"],
            ["NeuralColorPalette", f"{color_params:,}", f"color embed({N_COLORS},3)"],
            ["NeuralCompositor", f"{comp_params:,}", "Conv2d(3->32->32->3, residual)"],
            ["TOTAL", f"{total:,}", f"~{total * 4 / 1024:.1f} KB (float32)"],
        ],
        [24, 14, 48],
    )

    _row("Model file (V1):", f"{'FOUND' if v1_exists else 'NOT FOUND'} ({v1_size:,} bytes)" if v1_exists else "NOT FOUND")

    # V2 architecture
    try:
        from ncpu.neural.neural_terminal_renderer_v2 import (
            NeuralTerminalRendererV2, N_CHARS_V2, N_COLORS_V2,
        )
        _header("V2 Renderer Architecture")
        v2_renderer = NeuralTerminalRendererV2()
        v2_total = v2_renderer.count_params()
        v2_by_comp = v2_renderer.count_params_by_component()

        print()
        _row("Character set (V2):", f"{N_CHARS_V2} characters (extended Unicode)")
        _row("Color palette (V2):", f"{N_COLORS_V2} colors (xterm-256)")
        _row("Positional encoding:", "sinusoidal (y,x), 4 freq bands, 16-dim")
        print()
        _table(
            ["Component", "Parameters", "Description"],
            [
                ["NeuralGlyphGeneratorV2", f"{v2_by_comp['glyphs']:,}", f"char embed({N_CHARS_V2},64) + pos(16) + MLP(80->256->256->1)"],
                ["NeuralColorPaletteV2", f"{v2_by_comp['colors']:,}", f"color embed({N_COLORS_V2},3)"],
                ["NeuralCompositorV2", f"{v2_by_comp['compositor']:,}", "Conv2d(3->32->32->3, residual)"],
                ["TOTAL", f"{v2_total:,}", f"~{v2_total * 4 / 1024:.1f} KB (float32)"],
            ],
            [24, 14, 55],
        )
        _row("Model file (V2):", f"{'FOUND' if v2_exists else 'NOT FOUND'} ({v2_size:,} bytes)" if v2_exists else "NOT FOUND")
    except ImportError:
        print("\n  V2 renderer not available.")

    # Metal availability
    _header("Backend Availability")
    print()
    _row("PyTorch version:", torch.__version__)
    _row("MPS available:", "Yes" if torch.backends.mps.is_available() else "No")
    _row("CUDA available:", "Yes" if torch.cuda.is_available() else "No")

    metal_available = False
    try:
        from ncpu.neural.metal_neural_display import MetalNeuralDisplay
        if v1_exists:
            md = MetalNeuralDisplay(str(v1_path))
            metal_available = md.available
    except Exception:
        pass
    _row("Metal Neural Display:", "Available" if metal_available else "Not available")

    # Metal weight cache
    cache_path = models_dir / "terminal_renderer.metal_weights_full.npy"
    cache_base = models_dir / "terminal_renderer.metal_weights.npy"
    _row("Metal weight cache (full):", "FOUND" if cache_path.exists() else "Not found")
    _row("Metal weight cache (base):", "FOUND" if cache_base.exists() else "Not found")


# ---------------------------------------------------------------------------
# --benchmark: Speed test
# ---------------------------------------------------------------------------

def _benchmark_backend(
    name: str, display: NeuralDisplay, n_frames: int, terminal: TerminalState,
) -> float:
    """Benchmark a single backend, return average FPS."""
    # Warm up (3 frames)
    for _ in range(3):
        display.render()

    t0 = time.perf_counter()
    for _ in range(n_frames):
        display.render()
    elapsed = time.perf_counter() - t0

    fps = n_frames / elapsed if elapsed > 0 else 0
    ms_per_frame = (elapsed / n_frames) * 1000 if n_frames > 0 else 0
    return fps, ms_per_frame


def _fill_test_terminal(ts: TerminalState) -> None:
    """Fill a terminal state with realistic content for benchmarking."""
    ts.write_str("\x1b[2J\x1b[H")  # Clear
    ts.write_str("\x1b[1;32muser@ncpu\x1b[0m:\x1b[1;34m~/projects/nCPU\x1b[0m$ ")
    ts.write_str("ls -la --color=auto\r\n")
    ts.write_str("\x1b[1;34mdrwxr-xr-x\x1b[0m  12 user staff  384 Apr 12 10:30 \x1b[1;34m.\x1b[0m\r\n")
    ts.write_str("\x1b[1;34mdrwxr-xr-x\x1b[0m   5 user staff  160 Apr 10 09:15 \x1b[1;34m..\x1b[0m\r\n")
    ts.write_str("-rw-r--r--   1 user staff 1234 Apr 12 10:30 \x1b[0mREADME.md\x1b[0m\r\n")
    ts.write_str("-rwxr-xr-x   1 user staff 8192 Apr 11 14:22 \x1b[1;32mmain.py\x1b[0m\r\n")
    ts.write_str("\x1b[1;34mdrwxr-xr-x\x1b[0m   8 user staff  256 Apr 12 09:00 \x1b[1;34mncpu/\x1b[0m\r\n")
    ts.write_str("\x1b[1;34mdrwxr-xr-x\x1b[0m   3 user staff   96 Apr 11 16:45 \x1b[1;34mmodels/\x1b[0m\r\n")
    ts.write_str("-rw-r--r--   1 user staff  567 Apr 10 12:00 \x1b[0mpyproject.toml\x1b[0m\r\n")
    ts.write_str("\r\n\x1b[1;32muser@ncpu\x1b[0m:\x1b[1;34m~/projects/nCPU\x1b[0m$ ")
    # Fill some more rows
    for i in range(8, 20):
        ts.write_str(f"\x1b[{30 + (i % 8)}mLine {i}: The quick brown fox jumps over the lazy dog\x1b[0m\r\n")


def cmd_benchmark(n_frames: int = 100) -> None:
    """Run rendering benchmarks across available backends."""
    _header(f"Neural Terminal Benchmark ({n_frames} frames)")

    v1_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer.pt"
    if not v1_path.exists():
        print("\n  ERROR: V1 model not found at", v1_path)
        return

    results: list[list[str]] = []

    # Prepare test terminal state
    ts = TerminalState()
    _fill_test_terminal(ts)

    # --- PyTorch CPU ---
    print("\n  Benchmarking PyTorch CPU...", end="", flush=True)
    try:
        display_cpu = NeuralDisplay(str(v1_path), device="cpu")
        display_cpu.terminal = ts
        fps, ms = _benchmark_backend("PyTorch CPU", display_cpu, n_frames, ts)
        results.append(["PyTorch CPU", f"{fps:.1f}", f"{ms:.2f}", "V1"])
        print(f" {fps:.1f} FPS")
    except Exception as e:
        results.append(["PyTorch CPU", "ERROR", "-", str(e)[:40]])
        print(f" ERROR: {e}")

    # --- PyTorch MPS ---
    if torch.backends.mps.is_available():
        print("  Benchmarking PyTorch MPS...", end="", flush=True)
        try:
            display_mps = NeuralDisplay(str(v1_path), device="mps")
            display_mps.terminal = ts
            fps, ms = _benchmark_backend("PyTorch MPS", display_mps, n_frames, ts)
            results.append(["PyTorch MPS", f"{fps:.1f}", f"{ms:.2f}", "V1"])
            print(f" {fps:.1f} FPS")
        except Exception as e:
            results.append(["PyTorch MPS", "ERROR", "-", str(e)[:40]])
            print(f" ERROR: {e}")

    # --- PyTorch CUDA ---
    if torch.cuda.is_available():
        print("  Benchmarking PyTorch CUDA...", end="", flush=True)
        try:
            display_cuda = NeuralDisplay(str(v1_path), device="cuda")
            display_cuda.terminal = ts
            fps, ms = _benchmark_backend("PyTorch CUDA", display_cuda, n_frames, ts)
            results.append(["PyTorch CUDA", f"{fps:.1f}", f"{ms:.2f}", "V1"])
            print(f" {fps:.1f} FPS")
        except Exception as e:
            results.append(["PyTorch CUDA", "ERROR", "-", str(e)[:40]])
            print(f" ERROR: {e}")

    # --- Metal Neural Display (native shader, no PyTorch) ---
    try:
        from ncpu.neural.metal_neural_display import MetalNeuralDisplay
        md = MetalNeuralDisplay(str(v1_path))
        if md.available:
            print("  Benchmarking Metal GPU (native shader)...", end="", flush=True)
            # Warm up
            for _ in range(3):
                md.render(ts.chars, ts.fg, ts.bg, ts.cr, ts.cc)
            t0 = time.perf_counter()
            for _ in range(n_frames):
                md.render(ts.chars, ts.fg, ts.bg, ts.cr, ts.cc)
            elapsed = time.perf_counter() - t0
            fps = n_frames / elapsed
            ms = (elapsed / n_frames) * 1000
            results.append(["Metal GPU (native)", f"{fps:.1f}", f"{ms:.2f}", "V1"])
            print(f" {fps:.1f} FPS")
    except Exception as e:
        results.append(["Metal GPU (native)", "N/A", "-", str(e)[:40]])

    # --- V2 PyTorch (if available) ---
    v2_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer_v2.pt"
    if v2_path.exists():
        try:
            from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"  Benchmarking V2 PyTorch ({device})...", end="", flush=True)
            display_v2 = NeuralDisplayV2(str(v2_path), device=device)
            display_v2.terminal = ts
            fps, ms = _benchmark_backend(f"V2 PyTorch ({device})", display_v2, n_frames, ts)
            results.append([f"V2 PyTorch ({device})", f"{fps:.1f}", f"{ms:.2f}", "V2"])
            print(f" {fps:.1f} FPS")
        except Exception as e:
            results.append([f"V2 PyTorch", "ERROR", "-", str(e)[:40]])
            print(f" ERROR: {e}")

    # --- Results table ---
    _header("Benchmark Results")
    print()
    _table(
        ["Backend", "FPS", "ms/frame", "Model"],
        results,
        [24, 10, 12, 12],
    )

    # Find the best
    best_fps = 0.0
    best_name = ""
    for row in results:
        try:
            fps_val = float(row[1])
            if fps_val > best_fps:
                best_fps = fps_val
                best_name = row[0]
        except ValueError:
            pass

    if best_name:
        print(f"\n  Fastest: {best_name} at {best_fps:.1f} FPS")
        realtime = best_fps >= 30.0
        print(f"  Real-time (30+ FPS): {'YES' if realtime else 'NO'}")


# ---------------------------------------------------------------------------
# --palette: Show color palette
# ---------------------------------------------------------------------------

def cmd_palette() -> None:
    """Display the current ANSI color palette."""
    _header("ANSI 16-Color Palette")

    print()
    _table(
        ["Index", "Name", "R", "G", "B", "Hex"],
        [
            [
                str(i),
                ["Black", "Red", "Green", "Yellow", "Blue", "Magenta", "Cyan", "White",
                 "Bright Black", "Bright Red", "Bright Green", "Bright Yellow",
                 "Bright Blue", "Bright Magenta", "Bright Cyan", "Bright White"][i],
                str(ANSI_PALETTE[i][0]),
                str(ANSI_PALETTE[i][1]),
                str(ANSI_PALETTE[i][2]),
                f"#{ANSI_PALETTE[i][0]:02x}{ANSI_PALETTE[i][1]:02x}{ANSI_PALETTE[i][2]:02x}",
            ]
            for i in range(16)
        ],
        [8, 18, 6, 6, 6, 10],
    )

    # Show Metal palette if available
    try:
        from ncpu.neural.metal_neural_display import MetalNeuralDisplay
        v1_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer.pt"
        md = MetalNeuralDisplay(str(v1_path))
        if md.available:
            gpu_palette = md.get_palette()
            print("\n  Metal GPU palette (learned, may differ from ANSI defaults):")
            diffs = 0
            for i, (gpu, ansi) in enumerate(zip(gpu_palette, ANSI_PALETTE)):
                dr = abs(gpu[0] - ansi[0])
                dg = abs(gpu[1] - ansi[1])
                db = abs(gpu[2] - ansi[2])
                if dr + dg + db > 3:
                    print(f"    Color {i:2d}: ANSI=({ansi[0]:3d},{ansi[1]:3d},{ansi[2]:3d}) "
                          f"GPU=({gpu[0]:3d},{gpu[1]:3d},{gpu[2]:3d}) delta={dr+dg+db}")
                    diffs += 1
            if diffs == 0:
                print("    All colors match ANSI defaults (within +/-1 rounding).")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# --export-weights: Export for Metal shader
# ---------------------------------------------------------------------------

def cmd_export_weights(output_path: str) -> None:
    """Export model weights as numpy array for the Metal shader."""
    _header("Export Model Weights")

    v1_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer.pt"
    if not v1_path.exists():
        print(f"\n  ERROR: Model not found at {v1_path}")
        return

    print(f"\n  Loading model from {v1_path}...")
    state = torch.load(str(v1_path), map_location="cpu", weights_only=True)

    # Flatten all weights into a single float32 array
    weight_keys = [
        'glyphs.embed.weight', 'glyphs.net.0.weight', 'glyphs.net.0.bias',
        'glyphs.net.2.weight', 'glyphs.net.2.bias',
        'glyphs.net.4.weight', 'glyphs.net.4.bias',
        'colors.palette.weight',
    ]

    all_weights = []
    total = 0
    for key in weight_keys:
        if key in state:
            w = state[key].cpu().numpy().flatten()
            all_weights.append(w)
            _row(key, f"{w.shape[0]:,} floats")
            total += w.shape[0]
        else:
            print(f"  WARNING: Key '{key}' not found in checkpoint")

    # Check for compositor weights
    comp_keys = [
        'compositor.net.0.weight', 'compositor.net.0.bias',
        'compositor.net.2.weight', 'compositor.net.2.bias',
        'compositor.net.4.weight', 'compositor.net.4.bias',
    ]
    has_compositor = all(k in state for k in comp_keys)
    if has_compositor:
        print()
        print("  Compositor weights found, including in export:")
        for key in comp_keys:
            w = state[key].cpu().numpy().flatten()
            all_weights.append(w)
            _row(key, f"{w.shape[0]:,} floats")
            total += w.shape[0]

    weights_flat = np.concatenate(all_weights).astype(np.float32)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), weights_flat)

    print(f"\n  Exported {total:,} floats ({weights_flat.nbytes:,} bytes) to {out}")
    print(f"  Compositor: {'included' if has_compositor else 'not included'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Neural Terminal Configuration Utility"
    )
    parser.add_argument("--info", action="store_true", help="Show model info and system capabilities")
    parser.add_argument("--benchmark", action="store_true", help="Run rendering speed benchmark")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames for benchmark (default: 100)")
    parser.add_argument("--palette", action="store_true", help="Display color palette information")
    parser.add_argument("--export-weights", type=str, metavar="PATH", help="Export weights as numpy for Metal shader")
    args = parser.parse_args()

    # Default to --info if nothing specified
    if not any([args.info, args.benchmark, args.palette, args.export_weights]):
        args.info = True

    if args.info:
        cmd_info()

    if args.palette:
        cmd_palette()

    if args.benchmark:
        cmd_benchmark(args.frames)

    if args.export_weights:
        cmd_export_weights(args.export_weights)

    print()


if __name__ == "__main__":
    main()
