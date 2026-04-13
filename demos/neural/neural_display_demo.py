#!/usr/bin/env python3
"""Neural Display Demo -- basic neural terminal rendering showcase.

Demonstrates the nCPU neural rendering pipeline by rendering several text
scenes (welcome banner, system info, ANSI-colored code) through trained
neural networks and saving the results as PNG images.

Every pixel in the output is produced by neural network forward passes:
  char_code -> Embedding -> MLP -> 8x16 alpha mask
  color_code -> Embedding -> RGB
  alpha * fg + (1-a) * bg -> cell pixels
  assembled frame -> ConvNet -> refined frame

Usage:
    python demos/neural/neural_display_demo.py
    python demos/neural/neural_display_demo.py --output /tmp/demo.png
    python demos/neural/neural_display_demo.py --show-pipeline
    python demos/neural/neural_display_demo.py --device cpu
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
    NeuralTerminalRenderer,
    TerminalState,
    FRAME_H,
    FRAME_W,
    TERM_ROWS,
    TERM_COLS,
)

# ---------------------------------------------------------------------------
# ANSI escape helpers
# ---------------------------------------------------------------------------

ESC = "\033["

def sgr(code: int) -> str:
    """Return an SGR (Select Graphic Rendition) escape sequence."""
    return f"{ESC}{code}m"

def fg(color: int) -> str:
    """Foreground color (30-37 normal, 90-97 bright)."""
    return sgr(color)

def bg(color: int) -> str:
    """Background color (40-47 normal, 100-107 bright)."""
    return sgr(color)

def bold() -> str:
    return sgr(1)

def reset() -> str:
    return sgr(0)

def cursor_pos(row: int, col: int) -> str:
    """1-indexed cursor positioning."""
    return f"{ESC}{row};{col}H"


# ---------------------------------------------------------------------------
# Scene generators
# ---------------------------------------------------------------------------

def scene_welcome_banner() -> str:
    """Colorful welcome banner with box drawing and ANSI colors."""
    lines = []
    lines.append(f"{ESC}2J")  # clear screen
    lines.append(cursor_pos(1, 1))

    # Top border
    lines.append(f"{fg(36)}{bold()}+{'=' * 78}+{reset()}")

    # Title
    title = "nCPU Neural Display"
    subtitle = "Every Pixel Produced by Neural Networks"
    pad_t = (78 - len(title)) // 2
    pad_s = (78 - len(subtitle)) // 2
    lines.append(f"{fg(36)}|{' ' * pad_t}{fg(97)}{bold()}{title}{reset()}{fg(36)}{' ' * (78 - pad_t - len(title))}|{reset()}")
    lines.append(f"{fg(36)}|{' ' * pad_s}{fg(93)}{subtitle}{reset()}{fg(36)}{' ' * (78 - pad_s - len(subtitle))}|{reset()}")

    # Separator
    lines.append(f"{fg(36)}+{'-' * 78}+{reset()}")

    # Pipeline description
    desc = [
        f"  {fg(32)}char_code{reset()} -> {fg(33)}Embedding{reset()} -> {fg(35)}MLP{reset()} -> {fg(36)}alpha mask{reset()}",
        f"  {fg(32)}color_code{reset()} -> {fg(33)}Embedding{reset()} -> {fg(91)}R{reset()}{fg(92)}G{reset()}{fg(94)}B{reset()}",
        f"  {fg(36)}alpha{reset()} * {fg(91)}fg{reset()} + (1-a) * {fg(94)}bg{reset()} -> {fg(97)}cell pixels{reset()}",
        f"  {fg(33)}assembled grid{reset()} -> {fg(35)}ConvNet{reset()} -> {fg(97)}{bold()}refined frame{reset()}",
    ]
    for d in desc:
        lines.append(d)

    # Bottom border
    lines.append(f"{fg(36)}{bold()}+{'=' * 78}+{reset()}")

    # Color palette display
    lines.append("")
    lines.append(f"  {bold()}{fg(97)}ANSI Color Palette:{reset()}")
    palette_line = "  "
    color_names = [
        "Black", "Red", "Green", "Yellow",
        "Blue", "Magenta", "Cyan", "White",
    ]
    for i, name in enumerate(color_names):
        palette_line += f" {fg(30 + i)}{bg(40 + i)}  {reset()} {fg(30 + i)}{name:<8}{reset()}"
        if i == 3:
            lines.append(palette_line)
            palette_line = "  "
    lines.append(palette_line)

    # Bright colors
    lines.append("")
    palette_line = "  "
    for i in range(8):
        palette_line += f" {fg(90 + i)}{bg(100 + i)}  {reset()} {fg(90 + i)}{'Bright':<8}{reset()}"
        if i == 3:
            lines.append(palette_line)
            palette_line = "  "
    lines.append(palette_line)

    return "\n".join(lines)


def scene_system_info() -> str:
    """System information display with colored labels."""
    lines = []
    lines.append(f"{ESC}2J")
    lines.append(cursor_pos(1, 1))

    lines.append(f"{bold()}{fg(92)}nCPU System Information{reset()}")
    lines.append(f"{fg(36)}{'=' * 40}{reset()}")
    lines.append("")

    info_pairs = [
        ("Architecture", "Neural ARM64 (32-bit)"),
        ("ALU Models", "15 trained .pt networks"),
        ("neurOS Models", "11 trained .pt networks"),
        ("Display", "NeuralTerminalRenderer (143K params)"),
        ("Resolution", f"{FRAME_W}x{FRAME_H} ({TERM_COLS}x{TERM_ROWS} cells)"),
        ("Cell Size", "8x16 pixels"),
        ("Colors", "16 ANSI (V1) / 256 xterm (V2)"),
        ("GPU Backend", "Metal compute shaders (Rust)"),
        ("Execution", "neural / fast / compute modes"),
        ("Programs", "62 assembly (764 instructions)"),
    ]

    for label, value in info_pairs:
        lines.append(f"  {fg(33)}{label:<18}{reset()} {fg(97)}{value}{reset()}")

    lines.append("")
    lines.append(f"{fg(36)}{'=' * 40}{reset()}")
    lines.append("")

    # Memory map
    lines.append(f"{bold()}{fg(92)}Memory Layout{reset()}")
    lines.append(f"  {fg(91)}0x10000{reset()}  .text   {fg(90)}(code){reset()}")
    lines.append(f"  {fg(93)}0x50000{reset()}  .data   {fg(90)}(globals){reset()}")
    lines.append(f"  {fg(92)}0xFF000{reset()}  stack   {fg(90)}(grows down){reset()}")

    lines.append("")
    lines.append(f"  {fg(90)}All computation flows through trained neural networks.{reset()}")
    lines.append(f"  {fg(90)}Zero conventional logic gates. Zero lookup tables.{reset()}")

    return "\n".join(lines)


def scene_code_sample() -> str:
    """Syntax-highlighted code sample using ANSI escapes."""
    lines = []
    lines.append(f"{ESC}2J")
    lines.append(cursor_pos(1, 1))

    lines.append(f"{fg(90)}// Neural CPU — example program{reset()}")
    lines.append(f"{fg(90)}// Every instruction decoded by neural networks{reset()}")
    lines.append("")

    # C-like code with syntax highlighting
    code = [
        (f"{fg(35)}#include{reset()} {fg(31)}<ncpu.h>{reset()}", ""),
        ("", ""),
        (f"{fg(35)}int{reset()} {fg(93)}main{reset()}() {{", ""),
        (f"    {fg(35)}int{reset()} sum = {fg(36)}0{reset()};", ""),
        (f"    {fg(35)}int{reset()} n = {fg(36)}10{reset()};", ""),
        ("", ""),
        (f"    {fg(90)}// Neural ALU computes each addition{reset()}", ""),
        (f"    {fg(35)}for{reset()} ({fg(35)}int{reset()} i = {fg(36)}0{reset()}; i < n; i++) {{", ""),
        (f"        sum += i;  {fg(90)}// Kogge-Stone CLA{reset()}", ""),
        (f"    }}", ""),
        ("", ""),
        (f"    {fg(90)}// Neural display renders this output{reset()}", ""),
        (f"    {fg(93)}printf{reset()}({fg(31)}\"Sum: %d\\n\"{reset()}, sum);", ""),
        (f"    {fg(35)}return{reset()} {fg(36)}0{reset()};", ""),
        (f"}}", ""),
        ("", ""),
        (f"{fg(36)}+{'=' * 50}+{reset()}", ""),
        (f"{fg(36)}|{reset()}  {fg(92)}Output:{reset()} Sum: 45                                {fg(36)}|{reset()}", ""),
        (f"{fg(36)}|{reset()}  {fg(92)}ALU:{reset()}    Neural Kogge-Stone (8 passes)          {fg(36)}|{reset()}", ""),
        (f"{fg(36)}|{reset()}  {fg(92)}Branch:{reset()} Neural branch predictor                {fg(36)}|{reset()}", ""),
        (f"{fg(36)}|{reset()}  {fg(92)}Decode:{reset()} Neural LLM instruction decoder         {fg(36)}|{reset()}", ""),
        (f"{fg(36)}+{'=' * 50}+{reset()}", ""),
    ]

    for line, _ in code:
        lines.append(f"  {line}")

    return "\n".join(lines)


def scene_ascii_art() -> str:
    """ASCII art with color gradients."""
    lines = []
    lines.append(f"{ESC}2J")
    lines.append(cursor_pos(1, 1))

    # CPU chip ASCII art
    art = [
        f"  {fg(90)}         .----------------------------.{reset()}",
        f"  {fg(90)}        /   {fg(97)}{bold()}NEURAL PROCESSING UNIT{reset()}{fg(90)}    /{reset()}",
        f"  {fg(90)}       /                              /{reset()}",
        f"  {fg(90)}      /   {fg(36)}char -> embed -> MLP{reset()}{fg(90)}       /{reset()}",
        f"  {fg(90)}     /   {fg(33)}color -> embed -> RGB{reset()}{fg(90)}      /{reset()}",
        f"  {fg(90)}    /   {fg(35)}alpha x fg + bg -> px{reset()}{fg(90)}      /{reset()}",
        f"  {fg(90)}   /   {fg(32)}grid -> ConvNet -> frame{reset()}{fg(90)}   /{reset()}",
        f"  {fg(90)}  /                              /{reset()}",
        f"  {fg(90)} .------------------------------'{reset()}",
        "",
        f"  {fg(36)}+---------+---------+---------+---------+{reset()}",
        f"  {fg(36)}|{reset()} {fg(91)}Glyph{reset()}   {fg(36)}|{reset()} {fg(93)}Color{reset()}   {fg(36)}|{reset()} {fg(92)}Blend{reset()}   {fg(36)}|{reset()} {fg(95)}Comp{reset()}    {fg(36)}|{reset()}",
        f"  {fg(36)}|{reset()} {fg(91)}MLP{reset()}     {fg(36)}|{reset()} {fg(93)}Embed{reset()}   {fg(36)}|{reset()} {fg(92)}Alpha{reset()}   {fg(36)}|{reset()} {fg(95)}ConvNet{reset()} {fg(36)}|{reset()}",
        f"  {fg(36)}+---------+---------+---------+---------+{reset()}",
        "",
        f"  {fg(97)}143,251 parameters  |  566 KB  |  305 FPS (Metal){reset()}",
        "",
        f"  {fg(90)}Trained on bitmap font ground truth.{reset()}",
        f"  {fg(90)}Neural weights reproduce every glyph.{reset()}",
        f"  {fg(90)}No conventional rasterization anywhere in the pipeline.{reset()}",
        "",
        f"  {fg(33)}Model: models/display/terminal_renderer.pt{reset()}",
        f"  {fg(33)}Metal: kernels/rust_metal/src/neural_display.rs{reset()}",
    ]

    for line in art:
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline info
# ---------------------------------------------------------------------------

def print_pipeline_info(display: NeuralDisplay) -> None:
    """Print detailed architecture information about the neural display."""
    renderer = display.renderer

    total_params = renderer.count_params()
    glyph_params = sum(p.numel() for p in renderer.glyphs.parameters())
    color_params = sum(p.numel() for p in renderer.colors.parameters())
    comp_params = sum(p.numel() for p in renderer.compositor.parameters())

    print()
    print("=" * 60)
    print("  Neural Display Pipeline Architecture")
    print("=" * 60)
    print()
    print(f"  Total Parameters:      {total_params:>10,}")
    print(f"    NeuralGlyphGenerator: {glyph_params:>9,}")
    print(f"      Embedding:          {256 * 64:>9,}  (256 chars x 64-dim)")
    print(f"      FC1:                {64 * 256 + 256:>9,}  (64 -> 256)")
    print(f"      FC2:                {256 * 256 + 256:>9,}  (256 -> 256)")
    print(f"      FC3:                {256 * 128 + 128:>9,}  (256 -> 128)")
    print(f"    NeuralColorPalette:   {color_params:>9,}  (16 colors x 3 RGB)")
    print(f"    NeuralCompositor:     {comp_params:>9,}")
    print(f"      Conv1 (5x5):       {3 * 32 * 25 + 32:>9,}  (3 -> 32)")
    print(f"      Conv2 (3x3):       {32 * 32 * 9 + 32:>9,}  (32 -> 32)")
    print(f"      Conv3 (1x1):       {32 * 3 + 3:>9,}  (32 -> 3)")
    print()
    print(f"  Frame Resolution:      {FRAME_W} x {FRAME_H} ({TERM_COLS}x{TERM_ROWS} cells)")
    print(f"  Cell Size:             8 x 16 pixels")
    print(f"  Character Set:         256 (ASCII)")
    print(f"  Color Palette:         16 ANSI colors")
    print()
    print(f"  Metal GPU Available:   {display.metal_available}")
    print(f"  Device:                {display.device}")
    print()
    print("  Rendering Pipeline:")
    print("    1. char_code -> Embedding(256, 64) -> GELU MLP -> Sigmoid")
    print("       -> 8x16 alpha mask (learned glyph shape)")
    print("    2. color_code -> Embedding(16, 3) -> RGB float")
    print("    3. alpha * fg_rgb + (1-alpha) * bg_rgb -> cell pixels")
    print("    4. Tile cells into 640x384 frame")
    print("    5. (Optional) ConvNet compositor: anti-aliasing, refinement")
    print()
    print("=" * 60)


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def save_png(array: np.ndarray, path: Path) -> None:
    """Save an RGB numpy array as PNG using PIL."""
    try:
        from PIL import Image
    except ImportError:
        print(f"  [WARNING] PIL not installed -- cannot save PNG to {path}")
        print("  Install with: pip install Pillow")
        return
    img = Image.fromarray(array)
    img.save(str(path))
    print(f"  Saved: {path}")


def compose_grid(frames: list[np.ndarray], cols: int = 2) -> np.ndarray:
    """Arrange multiple frames in a grid with a 2-pixel dark border."""
    n = len(frames)
    rows = (n + cols - 1) // cols
    h, w, c = frames[0].shape
    border = 2
    grid_h = rows * h + (rows + 1) * border
    grid_w = cols * w + (cols + 1) * border
    canvas = np.full((grid_h, grid_w, c), 30, dtype=np.uint8)  # dark gray border

    for idx, frame in enumerate(frames):
        r, col_idx = divmod(idx, cols)
        y = border + r * (h + border)
        x = border + col_idx * (w + border)
        canvas[y:y + h, x:x + w] = frame

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Neural Display Demo -- render text through trained neural networks"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "models" / "display" / "neural_display_demo.png"),
        help="Output PNG path (default: models/display/neural_display_demo.png)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device: cpu, mps, cuda (default: auto-detect)",
    )
    parser.add_argument(
        "--show-pipeline", action="store_true",
        help="Print detailed pipeline architecture info",
    )
    args = parser.parse_args()

    model_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer.pt"

    print()
    print("=" * 60)
    print("  nCPU Neural Display Demo")
    print("  Rendering text through trained neural networks")
    print("=" * 60)
    print()
    print(f"  Model: {model_path}")

    display = NeuralDisplay(str(model_path), device=args.device)
    print(f"  Device: {display.device}")
    print(f"  Metal:  {display.metal_available}")
    print(f"  Params: {display.renderer.count_params():,}")
    print()

    if args.show_pipeline:
        print_pipeline_info(display)

    # Render scenes
    scenes = [
        ("Welcome Banner", scene_welcome_banner),
        ("System Info", scene_system_info),
        ("Code Sample", scene_code_sample),
        ("Architecture", scene_ascii_art),
    ]

    frames = []
    for name, gen in scenes:
        text = gen()
        t0 = time.perf_counter()
        frame = display.render_text(text)
        dt = time.perf_counter() - t0
        frames.append(frame)
        print(f"  Rendered '{name}': {frame.shape} in {dt * 1000:.1f} ms")

    # Save individual frames and composite
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    composite = compose_grid(frames, cols=2)
    save_png(composite, output_path)

    # Also save individual scenes
    stem = output_path.stem
    for i, (name, _) in enumerate(scenes):
        slug = name.lower().replace(" ", "_")
        individual_path = output_dir / f"{stem}_{slug}.png"
        save_png(frames[i], individual_path)

    print()
    print(f"  Composite: {composite.shape[1]}x{composite.shape[0]}")
    print("  Done.")
    print()


if __name__ == "__main__":
    main()
