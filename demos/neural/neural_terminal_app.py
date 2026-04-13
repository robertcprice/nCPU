#!/usr/bin/env python3
"""Neural Terminal Application — full-featured terminal with pygame display.

Every pixel on screen is produced by neural network forward passes: character
embeddings through an MLP generate glyph alpha masks, learned color palettes
provide RGB values, and alpha blending composites the final frame. No bitmap
fonts, no conventional rasterization.

Features:
  - Real-time neural rendering of typed characters at interactive FPS
  - Full ANSI color support (SGR codes, 16-color palette)
  - Cursor movement (arrow keys), scrolling, backspace, tab
  - FPS counter overlay showing neural rendering throughput
  - Configurable window scale and render device (CPU / MPS / CUDA)
  - V1 and V2 renderer selection

Usage:
    python demos/neural/neural_terminal_app.py
    python demos/neural/neural_terminal_app.py --scale 3
    python demos/neural/neural_terminal_app.py --v2 --device mps
    python demos/neural/neural_terminal_app.py --scale 1 --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import pygame
except ImportError:
    print("ERROR: pygame is required for the neural terminal app.")
    print("Install with: pip install pygame")
    sys.exit(1)

from ncpu.neural.neural_terminal_renderer import NeuralDisplay, FRAME_H, FRAME_W


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_TITLE = "nCPU Neural Terminal"
FPS_HISTORY = 60          # Rolling window for FPS averaging
TARGET_FPS = 60           # Pygame clock target
FPS_OVERLAY_COLOR = (0, 255, 0)
FPS_OVERLAY_BG = (0, 0, 0)

# Startup banner written into the terminal on launch
STARTUP_BANNER = (
    "\x1b[1;32mnCPU Neural Terminal\x1b[0m\r\n"
    "\x1b[36mEvery pixel rendered by neural networks.\x1b[0m\r\n"
    "\x1b[90m"
    "Type anything. Arrow keys, backspace, tab, Ctrl+C all work.\r\n"
    "Ctrl+Q or close window to exit.\x1b[0m\r\n"
    "\r\n"
    "\x1b[33m$ \x1b[0m"
)


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def _load_display(args: argparse.Namespace) -> NeuralDisplay:
    """Load the appropriate neural display (V1 or V2)."""
    if args.v2:
        from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2
        model_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer_v2.pt"
        return NeuralDisplayV2(str(model_path), device=args.device)
    else:
        model_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer.pt"
        return NeuralDisplay(str(model_path), device=args.device)


def _render_fps_overlay(surface: pygame.Surface, fps: float, scale: int):
    """Draw FPS counter in top-right corner."""
    font = pygame.font.SysFont("monospace", max(12, 10 * scale))
    label = f"{fps:.1f} FPS"
    text_surf = font.render(label, True, FPS_OVERLAY_COLOR, FPS_OVERLAY_BG)
    rect = text_surf.get_rect()
    rect.topright = (surface.get_width() - 4, 4)
    surface.blit(text_surf, rect)


# ---------------------------------------------------------------------------
# Keyboard mapping
# ---------------------------------------------------------------------------

def _key_to_bytes(event: pygame.event.Event) -> bytes:
    """Convert a pygame KEYDOWN event to the byte sequence a VT100 expects."""
    mods = event.mod

    # Ctrl+Q — quit signal (handled by caller)
    if event.key == pygame.K_q and (mods & pygame.KMOD_CTRL):
        return b'\x00'  # sentinel for quit

    # Ctrl+C
    if event.key == pygame.K_c and (mods & pygame.KMOD_CTRL):
        return b'^C\r\n'

    # Ctrl+L — clear screen
    if event.key == pygame.K_l and (mods & pygame.KMOD_CTRL):
        return b'\x1b[2J\x1b[H'

    # Arrow keys → ANSI escape sequences
    arrow_map = {
        pygame.K_UP:    b'\x1b[A',
        pygame.K_DOWN:  b'\x1b[B',
        pygame.K_RIGHT: b'\x1b[C',
        pygame.K_LEFT:  b'\x1b[D',
    }
    if event.key in arrow_map:
        return arrow_map[event.key]

    # Functional keys
    if event.key == pygame.K_RETURN:
        return b'\r\n'
    if event.key == pygame.K_BACKSPACE:
        return b'\x08 \x08'  # BS + erase + BS (visual backspace)
    if event.key == pygame.K_TAB:
        return b'\x09'
    if event.key == pygame.K_ESCAPE:
        return b'\x1b'
    if event.key == pygame.K_DELETE:
        return b'\x1b[3~'

    # Home/End
    if event.key == pygame.K_HOME:
        return b'\x1b[H'
    if event.key == pygame.K_END:
        return b'\x1b[F'

    # Printable text (pygame provides unicode)
    if event.unicode and ord(event.unicode) >= 0x20:
        return event.unicode.encode('utf-8')

    return b''


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Neural Terminal Application — every pixel is neural"
    )
    parser.add_argument(
        "--scale", type=int, default=2,
        help="Window scale multiplier (default: 2, window = 1280x768)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="PyTorch device for rendering (cpu, mps, cuda). Auto-detected if omitted."
    )
    parser.add_argument(
        "--v2", action="store_true",
        help="Use V2 renderer (positional encoding, 256-color palette)"
    )
    args = parser.parse_args()

    scale = max(1, args.scale)
    win_w = FRAME_W * scale
    win_h = FRAME_H * scale

    # --- Load neural display ---
    print(f"Loading neural terminal renderer ({'V2' if args.v2 else 'V1'})...")
    display = _load_display(args)
    backend = "Metal GPU" if getattr(display, 'metal_available', False) else "PyTorch"
    print(f"Render backend: {backend}, device: {getattr(display, 'device', 'n/a')}")
    print(f"Window: {win_w}x{win_h} (scale {scale}x)")

    # --- Write startup banner ---
    display.terminal.write_str(STARTUP_BANNER)

    # --- Initialize pygame ---
    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()
    pygame.key.set_repeat(400, 50)  # Key repeat: 400ms delay, 50ms interval

    # FPS tracking
    frame_times: deque[float] = deque(maxlen=FPS_HISTORY)
    running = True

    while running:
        t_start = time.perf_counter()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.KEYDOWN:
                data = _key_to_bytes(event)
                if data == b'\x00':  # Ctrl+Q sentinel
                    running = False
                    break
                if data:
                    display.write(data)

        if not running:
            break

        # --- Neural render ---
        frame = display.render()  # (384, 640, 3) uint8 numpy

        # --- Blit to pygame surface ---
        # pygame expects (width, height, 3) but surfarray wants (width, height)
        # so we transpose from (H, W, 3) to (W, H, 3) for surfarray
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        if scale != 1:
            surf = pygame.transform.scale(surf, (win_w, win_h))
        screen.blit(surf, (0, 0))

        # --- FPS overlay ---
        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > 1:
            avg_dt = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
        else:
            fps = 0.0
        _render_fps_overlay(screen, fps, scale)

        pygame.display.flip()
        clock.tick(TARGET_FPS)

    pygame.quit()
    print("Neural terminal closed.")


if __name__ == "__main__":
    main()
