#!/usr/bin/env python3
"""Neural Showcase -- multi-scene demonstration of neural terminal rendering.

Renders a sequence of diverse terminal scenes through the neural display
pipeline, capturing each as frames for an animated GIF. Scenes include
code rendering, ANSI art, system output, color palettes, and more.

Each scene is held for a configurable duration, with smooth transitions
between them (clear screen + redraw). The result showcases the full range
of the neural terminal renderer's capabilities.

Usage:
    python demos/neural/neural_showcase.py
    python demos/neural/neural_showcase.py --output /tmp/showcase.gif
    python demos/neural/neural_showcase.py --fps 12 --hold 2.0
    python demos/neural/neural_showcase.py --live
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
    TERM_ROWS,
    TERM_COLS,
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

def bg(c: int) -> str:
    return sgr(c)

def bold() -> str:
    return sgr(1)

def reset() -> str:
    return sgr(0)

def clear() -> str:
    return f"{ESC}2J{ESC}1;1H"


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

def scene_title() -> str:
    """Opening title card."""
    lines = [
        clear(),
        "",
        "",
        "",
        f"          {fg(36)}{bold()}+{'=' * 56}+{reset()}",
        f"          {fg(36)}|{reset()}                                                        {fg(36)}|{reset()}",
        f"          {fg(36)}|{reset()}     {fg(97)}{bold()}nCPU Neural Terminal Renderer{reset()}                {fg(36)}|{reset()}",
        f"          {fg(36)}|{reset()}     {fg(93)}Every Pixel From Neural Networks{reset()}             {fg(36)}|{reset()}",
        f"          {fg(36)}|{reset()}                                                        {fg(36)}|{reset()}",
        f"          {fg(36)}|{reset()}     {fg(90)}143,251 parameters  |  566 KB{reset()}              {fg(36)}|{reset()}",
        f"          {fg(36)}|{reset()}     {fg(90)}Char Embed -> MLP -> Alpha Mask{reset()}             {fg(36)}|{reset()}",
        f"          {fg(36)}|{reset()}     {fg(90)}Color Embed -> RGB -> Blend{reset()}                 {fg(36)}|{reset()}",
        f"          {fg(36)}|{reset()}                                                        {fg(36)}|{reset()}",
        f"          {fg(36)}{bold()}+{'=' * 56}+{reset()}",
        "",
        "",
        f"            {fg(90)}Robert Price  |  April 2026{reset()}",
    ]
    return "\n".join(lines)


def scene_color_palette() -> str:
    """Full ANSI color palette display."""
    lines = [clear()]
    lines.append(f"  {fg(97)}{bold()}ANSI Terminal Color Palette{reset()}")
    lines.append(f"  {fg(36)}{'=' * 50}{reset()}")
    lines.append("")

    # Normal colors with colored blocks
    lines.append(f"  {fg(97)}Standard Colors (30-37):{reset()}")
    row = "    "
    names = ["Black", "Red", "Green", "Yellow", "Blue", "Magenta", "Cyan", "White"]
    for i in range(8):
        row += f"{fg(30 + i)}{bg(40 + i)}  {names[i]:<8}{reset()} "
    lines.append(row)
    lines.append("")

    # Bright colors
    lines.append(f"  {fg(97)}Bright Colors (90-97):{reset()}")
    row = "    "
    for i in range(8):
        row += f"{fg(90 + i)}{bg(100 + i)}  {names[i]:<8}{reset()} "
    lines.append(row)
    lines.append("")

    # Color gradient bars
    lines.append(f"  {fg(97)}Gradient Bars:{reset()}")
    chars = " .:-=+*#%@"
    for color_base in [31, 32, 33, 34, 35, 36]:
        bar = f"    {fg(color_base)}"
        for ch in chars * 5:
            bar += ch
        bar += reset()
        lines.append(bar)

    lines.append("")

    # Character set sample
    lines.append(f"  {fg(97)}Character Rendering:{reset()}")
    lines.append(f"    {fg(92)}ABCDEFGHIJKLMNOPQRSTUVWXYZ{reset()}")
    lines.append(f"    {fg(93)}abcdefghijklmnopqrstuvwxyz{reset()}")
    lines.append(f"    {fg(91)}0123456789{reset()}")
    lines.append(f"    {fg(96)}!@#$%^&*()[]{{}}|;:',.<>?/{reset()}")

    return "\n".join(lines)


def scene_code_rendering() -> str:
    """Syntax-highlighted code sample."""
    lines = [clear()]
    lines.append(f"  {fg(90)}// Neural CPU -- sum computation via trained ALU{reset()}")
    lines.append("")
    lines.append(f"  {fg(35)}#include{reset()} {fg(31)}<stdio.h>{reset()}")
    lines.append("")
    lines.append(f"  {fg(35)}int{reset()} {fg(93)}main{reset()}() {{")
    lines.append(f"      {fg(35)}int{reset()} sum = {fg(36)}0{reset()};")
    lines.append(f"      {fg(35)}for{reset()} ({fg(35)}int{reset()} i = {fg(36)}1{reset()}; i <= {fg(36)}100{reset()}; i++) {{")
    lines.append(f"          sum += i;  {fg(90)}// Neural Kogge-Stone CLA{reset()}")
    lines.append(f"      }}")
    lines.append(f"      {fg(93)}printf{reset()}({fg(31)}\"Sum 1..100 = %d\\n\"{reset()}, sum);")
    lines.append(f"      {fg(35)}return{reset()} {fg(36)}0{reset()};")
    lines.append(f"  }}")
    lines.append("")
    lines.append(f"  {fg(36)}{'=' * 45}{reset()}")
    lines.append(f"  {fg(92)}Output:{reset()} Sum 1..100 = 5050")
    lines.append(f"  {fg(36)}{'=' * 45}{reset()}")
    lines.append("")
    lines.append(f"  {fg(33)}Instruction Decode:{reset()} Neural LLM (decode_llm)")
    lines.append(f"  {fg(33)}Addition:{reset()}           Kogge-Stone CLA (8 neural passes)")
    lines.append(f"  {fg(33)}Comparison:{reset()}         Neural subtraction + flags")
    lines.append(f"  {fg(33)}Branch:{reset()}             Neural branch predictor")
    lines.append(f"  {fg(33)}Memory:{reset()}             Neural pointer arithmetic")

    return "\n".join(lines)


def scene_system_output() -> str:
    """Simulated system output / shell session."""
    lines = [clear()]
    lines.append(f"{fg(92)}{bold()}ncpu{reset()}:{fg(94)}~{reset()}$ uname -a")
    lines.append(f"neurOS 1.0.0 ncpu0 ARM64 Neural-CPU")
    lines.append("")
    lines.append(f"{fg(92)}{bold()}ncpu{reset()}:{fg(94)}~{reset()}$ ncpu --status")
    lines.append(f"  {fg(97)}CPU Cores:{reset()}     1 (fully neural)")
    lines.append(f"  {fg(97)}ALU Models:{reset()}    15 active (.pt)")
    lines.append(f"  {fg(97)}OS Models:{reset()}     11 active (.pt)")
    lines.append(f"  {fg(97)}Display:{reset()}       terminal_renderer.pt (143K params)")
    lines.append(f"  {fg(97)}Uptime:{reset()}        Neural since boot")
    lines.append("")
    lines.append(f"{fg(92)}{bold()}ncpu{reset()}:{fg(94)}~{reset()}$ ncpu --benchmark --mode all")
    lines.append(f"  {fg(91)}neural-serial:{reset()}    2,389 IPS   {fg(90)}(step-by-step){reset()}")
    lines.append(f"  {fg(93)}woven-batch:{reset()}     33,000 IPS   {fg(90)}(vectorized){reset()}")
    lines.append(f"  {fg(92)}metal-add:{reset()}    1,500,000 IPS   {fg(90)}(GPU shader){reset()}")
    lines.append(f"  {fg(92)}metal-xor:{reset()}    8,000,000 IPS   {fg(90)}(GPU shader){reset()}")
    lines.append(f"  {fg(92)}metal-mul:{reset()}    4,000,000 IPS   {fg(90)}(GPU shader){reset()}")
    lines.append("")
    lines.append(f"{fg(92)}{bold()}ncpu{reset()}:{fg(94)}~{reset()}$ cat /proc/ncpu/display")
    lines.append(f"  Glyph MLP:    256-char embed -> 3-layer MLP -> 8x16 mask")
    lines.append(f"  Color Embed:  16 ANSI colors -> learned RGB")
    lines.append(f"  Compositor:   3-layer ConvNet (5x5 + 3x3 + 1x1)")
    lines.append(f"  Metal:        305 FPS (68.7 dB PSNR vs PyTorch)")
    lines.append("")
    lines.append(f"{fg(92)}{bold()}ncpu{reset()}:{fg(94)}~{reset()}$ _")

    return "\n".join(lines)


def scene_computation() -> str:
    """Mathematical computation display."""
    lines = [clear()]
    lines.append(f"  {fg(97)}{bold()}Neural Computation Trace{reset()}")
    lines.append(f"  {fg(36)}{'=' * 50}{reset()}")
    lines.append("")

    # Fibonacci
    lines.append(f"  {fg(33)}Fibonacci Sequence:{reset()}")
    a, b = 0, 1
    row = "    "
    for i in range(12):
        row += f"{fg(92)}{a}{reset()} "
        a, b = b, a + b
    lines.append(row)
    lines.append("")

    # Primes
    lines.append(f"  {fg(33)}Prime Sieve:{reset()}")
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    row = "    "
    for p in primes:
        row += f"{fg(91)}{p}{reset()} "
    lines.append(row)
    lines.append("")

    # Powers of 2
    lines.append(f"  {fg(33)}Powers of 2:{reset()}")
    row = "    "
    for i in range(16):
        row += f"{fg(94)}{1 << i}{reset()} "
    lines.append(row)
    lines.append("")

    # Binary
    lines.append(f"  {fg(33)}Binary Representation:{reset()}")
    for val in [42, 127, 255, 1024]:
        lines.append(f"    {fg(97)}{val:>5}{reset()} = {fg(36)}{val:016b}{reset()}")
    lines.append("")

    lines.append(f"  {fg(36)}{'=' * 50}{reset()}")
    lines.append(f"  {fg(90)}All arithmetic via trained neural ALU networks{reset()}")
    lines.append(f"  {fg(90)}Carry propagation: Kogge-Stone parallel prefix{reset()}")
    lines.append(f"  {fg(90)}Multiplication: Byte-pair LUT (256x256x16){reset()}")

    return "\n".join(lines)


def scene_closing() -> str:
    """Closing card."""
    lines = [clear()]
    lines.append("")
    lines.append("")
    lines.append("")
    lines.append(f"          {fg(36)}+{'=' * 56}+{reset()}")
    lines.append(f"          {fg(36)}|{reset()}                                                        {fg(36)}|{reset()}")
    lines.append(f"          {fg(36)}|{reset()}     {fg(97)}{bold()}nCPU: A Computer Made of Neural Networks{reset()}  {fg(36)}|{reset()}")
    lines.append(f"          {fg(36)}|{reset()}                                                        {fg(36)}|{reset()}")
    lines.append(f"          {fg(36)}|{reset()}     {fg(92)}15 ALU models{reset()}  trained .pt networks         {fg(36)}|{reset()}")
    lines.append(f"          {fg(36)}|{reset()}     {fg(93)}11 OS models{reset()}   neurOS kernel                {fg(36)}|{reset()}")
    lines.append(f"          {fg(36)}|{reset()}     {fg(91)} 1 display{reset()}     neural terminal renderer    {fg(36)}|{reset()}")
    lines.append(f"          {fg(36)}|{reset()}                                                        {fg(36)}|{reset()}")
    lines.append(f"          {fg(36)}|{reset()}     {fg(90)}Zero lookup tables. Zero bitmap fonts.{reset()}      {fg(36)}|{reset()}")
    lines.append(f"          {fg(36)}|{reset()}     {fg(90)}Every gate, every pixel: neural.{reset()}            {fg(36)}|{reset()}")
    lines.append(f"          {fg(36)}|{reset()}                                                        {fg(36)}|{reset()}")
    lines.append(f"          {fg(36)}+{'=' * 56}+{reset()}")
    lines.append("")
    lines.append("")
    lines.append(f"            {fg(90)}github.com/rp/nCPU  |  April 2026{reset()}")

    return "\n".join(lines)


ALL_SCENES = [
    ("Title", scene_title),
    ("Color Palette", scene_color_palette),
    ("Code Rendering", scene_code_rendering),
    ("System Output", scene_system_output),
    ("Computation", scene_computation),
    ("Closing", scene_closing),
]


# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------

def capture_scenes(
    display: NeuralDisplay,
    hold_seconds: float,
    fps: int,
) -> list[np.ndarray]:
    """Render all scenes and capture frames for each hold duration."""
    frames = []
    hold_frames = max(1, int(hold_seconds * fps))

    for name, gen in ALL_SCENES:
        text = gen()
        frame = display.render_text(text)
        print(f"    {name}: {hold_frames} frames")
        for _ in range(hold_frames):
            frames.append(frame.copy())

    return frames


# ---------------------------------------------------------------------------
# GIF / live display
# ---------------------------------------------------------------------------

def save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """Save frames as animated GIF."""
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


def run_live(display: NeuralDisplay, hold_seconds: float, scale: int) -> None:
    """Show the showcase in a live pygame window."""
    try:
        import pygame
    except ImportError:
        print()
        print("  [ERROR] pygame is required for --live mode.")
        print("  Install with: pip install pygame")
        print()
        sys.exit(1)

    pygame.init()
    win_w = FRAME_W * scale
    win_h = FRAME_H * scale
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("nCPU Neural Showcase")
    clock = pygame.time.Clock()

    running = True
    scene_idx = 0
    scene_start = time.perf_counter()

    # Render first scene
    text = ALL_SCENES[scene_idx][1]()
    frame = display.render_text(text)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Advance to next scene on space
                    scene_idx = (scene_idx + 1) % len(ALL_SCENES)
                    text = ALL_SCENES[scene_idx][1]()
                    frame = display.render_text(text)
                    scene_start = time.perf_counter()

        # Auto-advance
        elapsed = time.perf_counter() - scene_start
        if elapsed >= hold_seconds:
            scene_idx = (scene_idx + 1) % len(ALL_SCENES)
            text = ALL_SCENES[scene_idx][1]()
            frame = display.render_text(text)
            scene_start = time.perf_counter()

        surface = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2))
        )
        if scale != 1:
            surface = pygame.transform.scale(surface, (win_w, win_h))
        screen.blit(surface, (0, 0))

        # Scene indicator
        name = ALL_SCENES[scene_idx][0]
        pygame.display.set_caption(f"nCPU Neural Showcase -- {name}")

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Neural Showcase -- multi-scene neural terminal rendering demo"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "models" / "display" / "neural_showcase.gif"),
        help="Output GIF path",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device: cpu, mps, cuda (default: auto-detect)",
    )
    parser.add_argument(
        "--fps", type=int, default=12,
        help="Frames per second in the GIF (default: 12)",
    )
    parser.add_argument(
        "--hold", type=float, default=3.0,
        help="Seconds to hold each scene (default: 3.0)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Show in a live pygame window instead of saving GIF",
    )
    parser.add_argument(
        "--scale", type=int, default=2,
        help="Window scale for --live mode (default: 2)",
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
    print(f"  nCPU Neural Showcase ({version})")
    print("  Multi-scene neural terminal rendering demo")
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
    print(f"  Scenes: {len(ALL_SCENES)}")
    print()

    if args.live:
        print(f"  Mode: Live display (hold={args.hold}s, scale={args.scale}x)")
        print("  Press SPACE to advance, ESC/Q to quit.")
        print()
        run_live(display, args.hold, args.scale)
    else:
        print(f"  Mode: GIF capture (hold={args.hold}s, fps={args.fps})")
        print()
        print("  Rendering scenes:")
        t0 = time.perf_counter()
        frames = capture_scenes(display, args.hold, args.fps)
        dt = time.perf_counter() - t0
        print(f"  Total: {len(frames)} frames in {dt:.1f}s")
        print()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_gif(frames, output_path, args.fps)

    print()
    print("  Done.")
    print()


if __name__ == "__main__":
    main()
