#!/usr/bin/env python3
"""Neural Terminal Interactive Demo — typing + auto-demo mode.

Shows off the neural terminal rendering system in two ways:

  Interactive (default): Type freely and watch every keystroke get rendered
  through neural networks in real time. Characters go through embedding MLPs
  to produce glyph alpha masks, learned color palettes provide RGB values,
  and alpha blending composites the final 640x384 frame.

  Auto-demo (--demo): A pre-scripted session runs automatically, showcasing
  ANSI colors, cursor positioning, screen clearing, scrolling, and box-drawing
  characters — all rendered neurally.

Usage:
    python demos/neural/neural_terminal_interactive.py
    python demos/neural/neural_terminal_interactive.py --demo
    python demos/neural/neural_terminal_interactive.py --demo --scale 3
    python demos/neural/neural_terminal_interactive.py --device cpu --scale 1
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
    print("ERROR: pygame is required for the interactive neural terminal demo.")
    print("Install with: pip install pygame")
    sys.exit(1)

from ncpu.neural.neural_terminal_renderer import NeuralDisplay, FRAME_H, FRAME_W


# ---------------------------------------------------------------------------
# Demo script — a sequence of (text, delay_seconds) pairs
# ---------------------------------------------------------------------------

DEMO_SCRIPT: list[tuple[str, float]] = [
    # Clear screen and show title banner
    ("\x1b[2J\x1b[H", 0.3),
    ("\x1b[1;34m", 0.0),
    ("+----------------------------------------------+\r\n", 0.04),
    ("|   nCPU Neural Terminal - Interactive Demo     |\r\n", 0.04),
    ("|   Every pixel rendered by neural networks     |\r\n", 0.04),
    ("+----------------------------------------------+\r\n", 0.04),
    ("\x1b[0m\r\n", 0.5),

    # Show ANSI color palette
    ("\x1b[1;37mANSI Color Palette:\x1b[0m\r\n", 0.3),
    ("\x1b[30m Black \x1b[0m", 0.05),
    ("\x1b[31m Red \x1b[0m", 0.05),
    ("\x1b[32m Green \x1b[0m", 0.05),
    ("\x1b[33m Yellow \x1b[0m", 0.05),
    ("\x1b[34m Blue \x1b[0m", 0.05),
    ("\x1b[35m Magenta \x1b[0m", 0.05),
    ("\x1b[36m Cyan \x1b[0m", 0.05),
    ("\x1b[37m White \x1b[0m\r\n", 0.05),
    ("\x1b[90m Bright Black \x1b[0m", 0.05),
    ("\x1b[91m Bright Red \x1b[0m", 0.05),
    ("\x1b[92m Bright Green \x1b[0m", 0.05),
    ("\x1b[93m Bright Yellow \x1b[0m", 0.05),
    ("\x1b[94m Bright Blue \x1b[0m", 0.05),
    ("\x1b[95m Bright Magenta \x1b[0m", 0.05),
    ("\x1b[96m Bright Cyan \x1b[0m", 0.05),
    ("\x1b[97m Bright White \x1b[0m\r\n\r\n", 0.05),

    # Background colors
    ("\x1b[1;37mBackground Colors:\x1b[0m\r\n", 0.3),
    ("\x1b[40m  BG0  \x1b[0m", 0.04),
    ("\x1b[41m  BG1  \x1b[0m", 0.04),
    ("\x1b[42m  BG2  \x1b[0m", 0.04),
    ("\x1b[43m  BG3  \x1b[0m", 0.04),
    ("\x1b[44m  BG4  \x1b[0m", 0.04),
    ("\x1b[45m  BG5  \x1b[0m", 0.04),
    ("\x1b[46m  BG6  \x1b[0m", 0.04),
    ("\x1b[47m  BG7  \x1b[0m\r\n\r\n", 0.04),

    # Cursor positioning demo
    ("\x1b[1;33mCursor Positioning:\x1b[0m\r\n", 0.4),
    ("Writing at different positions...\r\n", 0.3),
    ("\x1b[17;5H\x1b[32m<-- row 17, col 5\x1b[0m", 0.5),
    ("\x1b[18;30H\x1b[35m<-- row 18, col 30\x1b[0m", 0.5),
    ("\x1b[19;55H\x1b[36m<-- row 19, col 55\x1b[0m", 0.5),

    # Progress bar animation
    ("\x1b[21;1H\x1b[1;37mProgress: \x1b[0m[\x1b[32m", 0.3),
    ("#", 0.05), ("#", 0.05), ("#", 0.05), ("#", 0.05), ("#", 0.05),
    ("#", 0.05), ("#", 0.05), ("#", 0.05), ("#", 0.05), ("#", 0.05),
    ("#", 0.05), ("#", 0.05), ("#", 0.05), ("#", 0.05), ("#", 0.05),
    ("#", 0.05), ("#", 0.05), ("#", 0.05), ("#", 0.05), ("#", 0.05),
    ("\x1b[0m] \x1b[1;32mDone!\x1b[0m\r\n", 0.5),

    # Final message
    ("\x1b[23;1H\x1b[1;96mNeural rendering complete. "
     "Close window or press Ctrl+Q.\x1b[0m", 1.0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_display(args: argparse.Namespace) -> NeuralDisplay:
    """Load display, V1 or V2."""
    if args.v2:
        from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2
        model_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer_v2.pt"
        return NeuralDisplayV2(str(model_path), device=args.device)
    model_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer.pt"
    return NeuralDisplay(str(model_path), device=args.device)


def _key_to_bytes(event: pygame.event.Event) -> bytes:
    """Convert pygame KEYDOWN to VT100 byte sequence."""
    mods = event.mod
    if event.key == pygame.K_q and (mods & pygame.KMOD_CTRL):
        return b'\x00'
    if event.key == pygame.K_c and (mods & pygame.KMOD_CTRL):
        return b'^C\r\n'
    if event.key == pygame.K_l and (mods & pygame.KMOD_CTRL):
        return b'\x1b[2J\x1b[H'
    arrow_map = {
        pygame.K_UP: b'\x1b[A', pygame.K_DOWN: b'\x1b[B',
        pygame.K_RIGHT: b'\x1b[C', pygame.K_LEFT: b'\x1b[D',
    }
    if event.key in arrow_map:
        return arrow_map[event.key]
    if event.key == pygame.K_RETURN:
        return b'\r\n'
    if event.key == pygame.K_BACKSPACE:
        return b'\x08 \x08'
    if event.key == pygame.K_TAB:
        return b'\x09'
    if event.unicode and ord(event.unicode) >= 0x20:
        return event.unicode.encode('utf-8')
    return b''


def _render_status(surface: pygame.Surface, text: str, scale: int):
    """Render a status line at bottom-left."""
    font = pygame.font.SysFont("monospace", max(12, 10 * scale))
    text_surf = font.render(text, True, (0, 255, 0), (0, 0, 0))
    rect = text_surf.get_rect()
    rect.bottomleft = (4, surface.get_height() - 4)
    surface.blit(text_surf, rect)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive neural terminal demo with auto-demo mode"
    )
    parser.add_argument("--scale", type=int, default=2, help="Window scale (default: 2)")
    parser.add_argument("--device", type=str, default=None, help="Render device")
    parser.add_argument("--v2", action="store_true", help="Use V2 renderer")
    parser.add_argument(
        "--demo", action="store_true",
        help="Run auto-demo showcasing ANSI features instead of interactive mode"
    )
    args = parser.parse_args()

    scale = max(1, args.scale)
    win_w = FRAME_W * scale
    win_h = FRAME_H * scale

    print(f"Loading neural renderer ({'V2' if args.v2 else 'V1'})...")
    display = _load_display(args)
    backend = "Metal GPU" if getattr(display, 'metal_available', False) else "PyTorch"
    print(f"Backend: {backend} | Window: {win_w}x{win_h}")

    pygame.init()
    title = "nCPU Neural Terminal - " + ("Auto Demo" if args.demo else "Interactive")
    pygame.display.set_caption(title)
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()
    pygame.key.set_repeat(400, 50)

    frame_times: deque[float] = deque(maxlen=60)
    running = True

    if args.demo:
        # --- Auto-demo mode ---
        script_idx = 0
        char_idx = 0
        last_char_time = time.perf_counter()
        script_text = ""
        script_delay = 0.0
        waiting = False
        wait_until = 0.0

        # Pre-load first entry
        if DEMO_SCRIPT:
            script_text, script_delay = DEMO_SCRIPT[0]

        while running:
            t0 = time.perf_counter()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    data = _key_to_bytes(event)
                    if data == b'\x00':
                        running = False
                        break

            if not running:
                break

            now = time.perf_counter()

            # Process demo script
            if script_idx < len(DEMO_SCRIPT):
                if waiting:
                    if now >= wait_until:
                        waiting = False
                        script_idx += 1
                        char_idx = 0
                        if script_idx < len(DEMO_SCRIPT):
                            script_text, script_delay = DEMO_SCRIPT[script_idx]
                else:
                    # Feed all characters of current entry at once (they arrive
                    # as ANSI sequences or text chunks, not per-char typing)
                    display.terminal.write_str(script_text)
                    char_idx = len(script_text)

                    # Wait for the specified delay before the next entry
                    waiting = True
                    wait_until = now + script_delay

            # Render
            frame = display.render()
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            if scale != 1:
                surf = pygame.transform.scale(surf, (win_w, win_h))
            screen.blit(surf, (0, 0))

            t1 = time.perf_counter()
            frame_times.append(t1 - t0)
            avg = sum(frame_times) / len(frame_times) if frame_times else 1.0
            fps = 1.0 / avg if avg > 0 else 0.0
            mode_str = "AUTO-DEMO" if script_idx < len(DEMO_SCRIPT) else "DONE"
            _render_status(screen, f"{fps:.1f} FPS | {mode_str}", scale)

            pygame.display.flip()
            clock.tick(TARGET_FPS)

    else:
        # --- Interactive mode ---
        display.terminal.write_str(
            "\x1b[1;32mnCPU Neural Terminal\x1b[0m\r\n"
            "\x1b[36mInteractive mode. Type anything.\x1b[0m\r\n"
            "\x1b[90mCtrl+Q to quit, Ctrl+L to clear.\x1b[0m\r\n\r\n"
            "\x1b[33m$ \x1b[0m"
        )

        while running:
            t0 = time.perf_counter()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    data = _key_to_bytes(event)
                    if data == b'\x00':
                        running = False
                        break
                    if data:
                        display.write(data)

            if not running:
                break

            frame = display.render()
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            if scale != 1:
                surf = pygame.transform.scale(surf, (win_w, win_h))
            screen.blit(surf, (0, 0))

            t1 = time.perf_counter()
            frame_times.append(t1 - t0)
            avg = sum(frame_times) / len(frame_times) if frame_times else 1.0
            fps = 1.0 / avg if avg > 0 else 0.0
            _render_status(screen, f"{fps:.1f} FPS | INTERACTIVE", scale)

            pygame.display.flip()
            clock.tick(TARGET_FPS)

    pygame.quit()
    print("Interactive demo closed.")


TARGET_FPS = 60

if __name__ == "__main__":
    main()
