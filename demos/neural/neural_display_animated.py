#!/usr/bin/env python3
"""Neural Display Animated -- animated GIF of neural terminal rendering.

Creates an animated GIF showing the neural rendering pipeline in action.
Default mode types text character-by-character (typewriter effect) and
captures every frame through the neural renderer.

Modes:
  - Default: typewriter effect, typing a multi-scene terminal session
  - --fast: quick Fibonacci sequence computation demo
  - --program <path>: render output of a custom C source file line by line

Every frame passes through: char embedding -> MLP -> alpha mask -> color
embedding -> RGB -> alpha blend -> frame assembly.

Usage:
    python demos/neural/neural_display_animated.py
    python demos/neural/neural_display_animated.py --fast
    python demos/neural/neural_display_animated.py --fps 20 --output /tmp/anim.gif
    python demos/neural/neural_display_animated.py --program programs/hello.c
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.neural.neural_terminal_renderer import (
    NeuralDisplay,
    FRAME_H,
    FRAME_W,
)
from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

ESC = "\033["

def sgr(code: int) -> str:
    return f"{ESC}{code}m"

def fg(c: int) -> str:
    return sgr(c)

def bold() -> str:
    return sgr(1)

def reset() -> str:
    return sgr(0)


# ---------------------------------------------------------------------------
# Typewriter content generators
# ---------------------------------------------------------------------------

def typewriter_lines_default() -> list[str]:
    """Return lines for the default typewriter demo."""
    return [
        f"{fg(92)}{bold()}nCPU Neural Display{reset()}",
        f"{fg(36)}{'=' * 40}{reset()}",
        "",
        f"{fg(33)}$>{reset()} cat /proc/ncpu/info",
        f"  {fg(97)}Architecture:{reset()} Neural ARM64",
        f"  {fg(97)}ALU Models:{reset()}  15 trained networks",
        f"  {fg(97)}Display:{reset()}     NeuralTerminalRenderer",
        f"  {fg(97)}Parameters:{reset()} 143,251",
        "",
        f"{fg(33)}$>{reset()} ncpu --run hello.asm",
        f"  {fg(90)}[decode]{reset()} MOV X0, #72   {fg(90)}// 'H'{reset()}",
        f"  {fg(90)}[alu]{reset()}    ADD X1, X0, #29 {fg(90)}// 'e'{reset()}",
        f"  {fg(90)}[mem]{reset()}    STR X0, [SP]",
        f"  {fg(92)}Hello, World!{reset()}",
        "",
        f"{fg(33)}$>{reset()} ncpu --benchmark",
        f"  {fg(91)}neural-serial:{reset()}   2,389 IPS",
        f"  {fg(93)}woven-batch:{reset()}    33,000 IPS",
        f"  {fg(92)}metal-neural:{reset()} 1,500,000 IPS",
        "",
        f"{fg(36)}{'=' * 40}{reset()}",
        f"{fg(90)}Every pixel rendered by neural networks.{reset()}",
    ]


def typewriter_lines_fibonacci() -> list[str]:
    """Return lines for the fast Fibonacci demo."""
    lines = [
        f"{fg(92)}{bold()}Fibonacci Sequence{reset()}",
        f"{fg(36)}Neural ALU computes each step{reset()}",
        "",
    ]
    a, b = 0, 1
    for i in range(16):
        lines.append(f"  {fg(33)}fib({i:>2}){reset()} = {fg(97)}{a}{reset()}")
        a, b = b, a + b
    lines.append("")
    lines.append(f"{fg(90)}Additions via Kogge-Stone CLA network{reset()}")
    return lines


def typewriter_lines_from_file(path: Path) -> list[str]:
    """Read a file and return its lines with basic syntax coloring."""
    if not path.exists():
        return [f"{fg(91)}Error: {path} not found{reset()}"]
    text = path.read_text()
    colored = []
    colored.append(f"{fg(33)}$>{reset()} cat {path.name}")
    colored.append(f"{fg(36)}{'-' * 40}{reset()}")
    for line in text.splitlines()[:20]:  # cap at 20 lines for terminal
        # Basic C syntax highlighting
        stripped = line
        if stripped.lstrip().startswith("//"):
            colored.append(f"  {fg(90)}{stripped}{reset()}")
        elif stripped.lstrip().startswith("#"):
            colored.append(f"  {fg(35)}{stripped}{reset()}")
        else:
            colored.append(f"  {fg(97)}{stripped}{reset()}")
    colored.append(f"{fg(36)}{'-' * 40}{reset()}")
    return colored


# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------

def capture_typewriter(
    display: NeuralDisplay,
    lines: list[str],
    chars_per_frame: int = 3,
    hold_frames: int = 8,
) -> list[np.ndarray]:
    """Capture frames as text is typed character by character.

    Args:
        display: Neural display instance.
        lines: Lines of text to type (may contain ANSI escapes).
        chars_per_frame: How many characters to advance per captured frame.
        hold_frames: Number of extra frames to hold at the end.

    Returns:
        List of (384, 640, 3) uint8 numpy arrays.
    """
    frames = []
    display.reset()

    # Build full text
    full_text = "\n".join(lines)
    encoded = full_text.encode("utf-8")

    # Capture initial blank frame
    frames.append(display.render())

    # Type character by character, capturing at intervals
    byte_idx = 0
    frame_counter = 0
    for b in encoded:
        display.terminal.write(bytes([b]))
        byte_idx += 1
        frame_counter += 1
        if frame_counter >= chars_per_frame:
            frames.append(display.render())
            frame_counter = 0

    # Capture final state if there are remaining characters
    if frame_counter > 0:
        frames.append(display.render())

    # Hold final frame
    final = frames[-1]
    for _ in range(hold_frames):
        frames.append(final.copy())

    return frames


# ---------------------------------------------------------------------------
# GIF saving
# ---------------------------------------------------------------------------

def save_gif(frames: list[np.ndarray], path: Path, fps: int = 15) -> None:
    """Save frames as an animated GIF using PIL."""
    try:
        from PIL import Image
    except ImportError:
        print(f"  [WARNING] PIL not installed -- cannot save GIF to {path}")
        print("  Install with: pip install Pillow")
        return

    images = [Image.fromarray(f) for f in frames]
    duration = max(1, int(1000 / fps))
    images[0].save(
        str(path),
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    size_kb = path.stat().st_size / 1024
    print(f"  Saved: {path} ({len(frames)} frames, {size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Neural Display Animated -- create GIF of neural terminal rendering"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "models" / "display" / "neural_display_animated.gif"),
        help="Output GIF path",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device: cpu, mps, cuda (default: auto-detect)",
    )
    parser.add_argument(
        "--fps", type=int, default=15,
        help="Frames per second in the GIF (default: 15)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick Fibonacci demo instead of full typewriter",
    )
    parser.add_argument(
        "--program", type=str, default=None,
        help="Path to a C source file to render",
    )
    parser.add_argument(
        "--chars-per-frame", type=int, default=3,
        help="Characters to advance per captured frame (default: 3)",
    )
    parser.add_argument(
        "--v2", action="store_true",
        help="Use V2 renderer (512-wide MLP, 1024 chars, 256 colors)",
    )
    args = parser.parse_args()

    if args.v2:
        model_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer_v2.pt"
        version = "V2"
    else:
        model_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer.pt"
        version = "V1"

    print()
    print("=" * 60)
    print(f"  nCPU Neural Display -- Animated GIF Generator ({version})")
    print("=" * 60)
    print()
    print(f"  Model: {model_path}")

    if args.v2:
        display = NeuralDisplayV2(str(model_path), device=args.device)
    else:
        display = NeuralDisplay(str(model_path), device=args.device)
    print(f"  Device: {display.device}")
    if hasattr(display, 'metal_available'):
        print(f"  Metal:  {display.metal_available}")
    print()

    # Select content
    if args.program:
        lines = typewriter_lines_from_file(Path(args.program))
        mode_name = f"Program: {args.program}"
    elif args.fast:
        lines = typewriter_lines_fibonacci()
        mode_name = "Fibonacci (fast)"
    else:
        lines = typewriter_lines_default()
        mode_name = "Default typewriter"

    print(f"  Mode: {mode_name}")
    print(f"  Lines: {len(lines)}")
    print(f"  FPS: {args.fps}")
    print(f"  Chars/frame: {args.chars_per_frame}")
    print()

    # Capture frames
    print("  Capturing frames...")
    t0 = time.perf_counter()
    frames = capture_typewriter(
        display, lines,
        chars_per_frame=args.chars_per_frame,
        hold_frames=max(5, args.fps),  # hold for ~1 second
    )
    dt = time.perf_counter() - t0
    print(f"  Captured {len(frames)} frames in {dt:.1f}s ({len(frames) / dt:.1f} frames/s)")

    # Save GIF
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, output_path, fps=args.fps)

    print()
    print("  Done.")
    print()


if __name__ == "__main__":
    main()
