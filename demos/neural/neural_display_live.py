#!/usr/bin/env python3
"""Neural Display Live -- real-time interactive neural terminal rendering.

Opens a pygame window showing the neural terminal renderer in real-time.
User keystrokes are fed through the VT100 state machine and rendered by
neural networks on every frame.

Modes:
  - Default: interactive shell -- type and see neural rendering live
  - --text: non-interactive text demo that types predefined content
  - --program <path>: display output of a C source file

Every pixel on screen is produced by neural network forward passes.

Usage:
    python demos/neural/neural_display_live.py
    python demos/neural/neural_display_live.py --scale 2
    python demos/neural/neural_display_live.py --text
    python demos/neural/neural_display_live.py --program programs/hello.c
    python demos/neural/neural_display_live.py --device cpu
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
# Text demo content
# ---------------------------------------------------------------------------

def demo_text_lines() -> list[str]:
    """Lines for the non-interactive text demo."""
    return [
        f"{fg(92)}{bold()}nCPU Neural Terminal{reset()} -- Live Rendering",
        f"{fg(36)}{'=' * 50}{reset()}",
        "",
        f"  {fg(97)}This text is rendered entirely by neural networks.{reset()}",
        f"  {fg(97)}No bitmap fonts, no lookup tables, no rasterization.{reset()}",
        "",
        f"  {fg(33)}Pipeline:{reset()}",
        f"    {fg(35)}1.{reset()} Character code -> Embedding(256, 64)",
        f"    {fg(35)}2.{reset()} Embedding -> MLP(64->256->256->128) -> alpha mask",
        f"    {fg(35)}3.{reset()} Color code -> Embedding(16, 3) -> RGB",
        f"    {fg(35)}4.{reset()} alpha * fg + (1-a) * bg -> cell pixels",
        f"    {fg(35)}5.{reset()} Tile 1920 cells -> 640x384 frame",
        "",
        f"  {fg(91)}Red{reset()} {fg(92)}Green{reset()} {fg(93)}Yellow{reset()} {fg(94)}Blue{reset()} {fg(95)}Magenta{reset()} {fg(96)}Cyan{reset()} {fg(97)}White{reset()}",
        "",
        f"  {fg(33)}Characters:{reset()} ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        f"  {fg(33)}Numbers:{reset()}    0123456789",
        f"  {fg(33)}Symbols:{reset()}    !@#$%^&*()+-=[]{{}}|;:',.<>?/",
        "",
        f"{fg(36)}{'=' * 50}{reset()}",
        f"  {fg(90)}143,251 parameters | 566 KB | ~305 FPS on Metal{reset()}",
    ]


# ---------------------------------------------------------------------------
# Pygame key mapping
# ---------------------------------------------------------------------------

def pygame_key_to_bytes(event) -> bytes:
    """Convert a pygame KEYDOWN event to terminal bytes.

    Handles printable characters, Enter, Backspace, Tab, Escape, and
    arrow keys (as VT100 escape sequences).
    """
    import pygame

    key = event.key
    mods = event.mod

    # Arrow keys -> VT100 escape sequences
    if key == pygame.K_UP:
        return b"\033[A"
    elif key == pygame.K_DOWN:
        return b"\033[B"
    elif key == pygame.K_RIGHT:
        return b"\033[C"
    elif key == pygame.K_LEFT:
        return b"\033[D"
    elif key == pygame.K_RETURN:
        return b"\n"
    elif key == pygame.K_BACKSPACE:
        return b"\x08"
    elif key == pygame.K_TAB:
        return b"\t"
    elif key == pygame.K_ESCAPE:
        return b"\x1b"
    elif key == pygame.K_DELETE:
        return b"\033[3~"

    # Printable characters via event.unicode
    if event.unicode and ord(event.unicode) >= 0x20:
        return event.unicode.encode("utf-8")

    return b""


# ---------------------------------------------------------------------------
# Live display loop
# ---------------------------------------------------------------------------

def run_interactive(display: NeuralDisplay, scale: int) -> None:
    """Run the interactive pygame display loop."""
    try:
        import pygame
    except ImportError:
        print()
        print("  [ERROR] pygame is required for live display.")
        print("  Install with: pip install pygame")
        print()
        sys.exit(1)

    pygame.init()
    win_w = FRAME_W * scale
    win_h = FRAME_H * scale
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("nCPU Neural Display -- Interactive")
    clock = pygame.time.Clock()

    # Write initial prompt
    display.reset()
    prompt = f"{fg(92)}ncpu{reset()}:{fg(94)}~{reset()}{fg(97)}$ {reset()}"
    display.terminal.write_str(prompt)

    running = True
    needs_render = True
    frame_count = 0
    fps_timer = time.perf_counter()
    display_fps = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Ctrl+C or Ctrl+Q to quit
                if event.key == pygame.K_c and (event.mod & pygame.KMOD_CTRL):
                    running = False
                    continue
                if event.key == pygame.K_q and (event.mod & pygame.KMOD_CTRL):
                    running = False
                    continue

                data = pygame_key_to_bytes(event)
                if data:
                    if data == b"\n":
                        # Echo newline and re-prompt
                        display.terminal.write_str("\n")
                        display.terminal.write_str(prompt)
                    elif data == b"\x08":
                        # Backspace: move cursor back and erase
                        if display.terminal.cc > 0:
                            display.terminal.write(b"\x08")
                            display.terminal.write_str(" ")
                            display.terminal.write(b"\x08")
                    else:
                        display.terminal.write(data)
                    needs_render = True

        # Render neural frame
        if needs_render:
            frame = display.render()
            # Convert to pygame surface
            surface = pygame.surfarray.make_surface(
                np.transpose(frame, (1, 0, 2))
            )
            if scale != 1:
                surface = pygame.transform.scale(surface, (win_w, win_h))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            needs_render = False

            frame_count += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                display_fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now
                pygame.display.set_caption(
                    f"nCPU Neural Display -- {display_fps:.1f} FPS"
                )

        clock.tick(60)  # cap at 60 FPS

    pygame.quit()


def run_text_demo(display: NeuralDisplay, scale: int) -> None:
    """Run the non-interactive text demo in a pygame window."""
    try:
        import pygame
    except ImportError:
        print()
        print("  [ERROR] pygame is required for live display.")
        print("  Install with: pip install pygame")
        print()
        sys.exit(1)

    pygame.init()
    win_w = FRAME_W * scale
    win_h = FRAME_H * scale
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("nCPU Neural Display -- Text Demo")
    clock = pygame.time.Clock()

    lines = demo_text_lines()
    full_text = "\n".join(lines)
    encoded = full_text.encode("utf-8")

    display.reset()
    byte_idx = 0
    running = True
    typing_done = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        # Type characters
        if not typing_done and byte_idx < len(encoded):
            # Feed a few bytes per frame for smooth typewriter effect
            chunk_size = min(4, len(encoded) - byte_idx)
            display.terminal.write(encoded[byte_idx:byte_idx + chunk_size])
            byte_idx += chunk_size
        elif not typing_done:
            typing_done = True

        # Render
        frame = display.render()
        surface = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2))
        )
        if scale != 1:
            surface = pygame.transform.scale(surface, (win_w, win_h))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def run_program_demo(display: NeuralDisplay, program_path: Path, scale: int) -> None:
    """Display the contents of a program file through neural rendering."""
    try:
        import pygame
    except ImportError:
        print()
        print("  [ERROR] pygame is required for live display.")
        print("  Install with: pip install pygame")
        print()
        sys.exit(1)

    if not program_path.exists():
        print(f"  [ERROR] File not found: {program_path}")
        sys.exit(1)

    pygame.init()
    win_w = FRAME_W * scale
    win_h = FRAME_H * scale
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"nCPU Neural Display -- {program_path.name}")
    clock = pygame.time.Clock()

    # Prepare colored content
    source = program_path.read_text()
    header = f"{fg(92)}{bold()}{program_path.name}{reset()}\n"
    header += f"{fg(36)}{'-' * 50}{reset()}\n"

    colored_lines = []
    for line in source.splitlines()[:TERM_ROWS - 4]:
        stripped = line.lstrip()
        if stripped.startswith("//"):
            colored_lines.append(f"  {fg(90)}{line}{reset()}")
        elif stripped.startswith("#"):
            colored_lines.append(f"  {fg(35)}{line}{reset()}")
        elif any(kw in stripped for kw in ["int ", "void ", "char ", "return ", "for ", "if ", "while "]):
            colored_lines.append(f"  {fg(33)}{line}{reset()}")
        else:
            colored_lines.append(f"  {fg(97)}{line}{reset()}")

    full_text = header + "\n".join(colored_lines)
    encoded = full_text.encode("utf-8")

    display.reset()
    byte_idx = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        if byte_idx < len(encoded):
            chunk = min(6, len(encoded) - byte_idx)
            display.terminal.write(encoded[byte_idx:byte_idx + chunk])
            byte_idx += chunk

        frame = display.render()
        surface = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2))
        )
        if scale != 1:
            surface = pygame.transform.scale(surface, (win_w, win_h))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Neural Display Live -- real-time interactive neural terminal rendering"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device: cpu, mps, cuda (default: auto-detect)",
    )
    parser.add_argument(
        "--scale", type=int, default=2,
        help="Window scale factor (default: 2, so 1280x768)",
    )
    parser.add_argument(
        "--text", action="store_true",
        help="Non-interactive text demo mode",
    )
    parser.add_argument(
        "--program", type=str, default=None,
        help="Path to a C source file to display",
    )
    args = parser.parse_args()

    model_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer.pt"

    print()
    print("=" * 60)
    print("  nCPU Neural Display -- Live Rendering")
    print("=" * 60)
    print()
    print(f"  Model: {model_path}")

    display = NeuralDisplay(str(model_path), device=args.device)
    print(f"  Device: {display.device}")
    print(f"  Metal:  {display.metal_available}")
    print(f"  Scale:  {args.scale}x ({FRAME_W * args.scale}x{FRAME_H * args.scale})")
    print()

    if args.program:
        print(f"  Mode: Program viewer ({args.program})")
        print("  Press ESC or Q to quit.")
        print()
        run_program_demo(display, Path(args.program), args.scale)
    elif args.text:
        print("  Mode: Text demo (non-interactive)")
        print("  Press ESC or Q to quit.")
        print()
        run_text_demo(display, args.scale)
    else:
        print("  Mode: Interactive shell")
        print("  Type to see neural rendering in real-time.")
        print("  Ctrl+C or Ctrl+Q to quit.")
        print()
        run_interactive(display, args.scale)

    print("  Done.")
    print()


if __name__ == "__main__":
    main()
