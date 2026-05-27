#!/usr/bin/env python3
"""Neural Terminal PTY — a real shell rendered through neural networks.

This is the flagship demo of the nCPU neural display system. It spawns a real
shell process (bash, zsh, or user-specified) attached to a pseudo-terminal,
then renders every byte of shell output through the neural rendering pipeline.

The result is a fully functional terminal emulator where every single pixel is
produced by neural network forward passes: character embeddings generate glyph
masks via MLPs, a learned color palette provides RGB values, and alpha blending
composites the 640x384 frame. No bitmap fonts. No conventional rasterization.

You can run vim, htop, ls with colors, compile code — everything a normal
terminal does, but every frame is neurally rendered.

Features:
  - Spawns real shell via pty.fork() with proper TIOCSWINSZ (24x80)
  - Non-blocking reads from PTY via select()
  - Full keyboard forwarding including Ctrl sequences and function keys
  - Graceful shell lifecycle: SIGHUP on close, SIGKILL fallback
  - TERM=xterm-256color for maximum compatibility
  - Configurable shell, scale, and device

Usage:
    python demos/neural/neural_terminal_pty.py
    python demos/neural/neural_terminal_pty.py --shell /bin/bash
    python demos/neural/neural_terminal_pty.py --scale 3 --device mps
    python demos/neural/neural_terminal_pty.py --v2
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import time
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import pygame
except ImportError:
    print("ERROR: pygame is required for the neural PTY terminal.")
    print("Install with: pip install pygame")
    sys.exit(1)

from ncpu.neural.neural_terminal_renderer import (
    NeuralDisplay, FRAME_H, FRAME_W, TERM_ROWS, TERM_COLS,
)


# ---------------------------------------------------------------------------
# PTY management
# ---------------------------------------------------------------------------

class ShellProcess:
    """Manages a shell child process attached to a PTY."""

    def __init__(
        self,
        shell: str,
        rows: int = TERM_ROWS,
        cols: int = TERM_COLS,
        *,
        login: bool = True,
        inherit_env: bool = True,
        extra_env: dict[str, str] | None = None,
    ):
        self.shell = shell
        self.rows = rows
        self.cols = cols
        self.login = login
        self.inherit_env = inherit_env
        self.extra_env = dict(extra_env or {})
        self.pid: int = -1
        self.fd: int = -1
        self._alive = False

    def start(self) -> None:
        """Fork a child process with a PTY."""
        pid, fd = pty.fork()

        if pid == 0:
            # Child process — exec the shell
            env = dict(os.environ) if self.inherit_env else {}
            env["TERM"] = "xterm-256color"
            env["COLUMNS"] = str(self.cols)
            env["LINES"] = str(self.rows)
            env.setdefault("LANG", "en_US.UTF-8")
            env.update(self.extra_env)
            argv = [self.shell]
            if self.login:
                argv.append("--login")
            os.execvpe(self.shell, argv, env)
            # execvp does not return on success
            sys.exit(1)

        # Parent process
        self.pid = pid
        self.fd = fd
        self._alive = True

        # Set window size on the PTY
        winsize = struct.pack("HHHH", self.rows, self.cols, 0, 0)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

        # Make reads non-blocking
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def read(self, max_bytes: int = 4096) -> bytes:
        """Non-blocking read from the PTY. Returns empty bytes if nothing."""
        if not self._alive:
            return b''
        try:
            rlist, _, _ = select.select([self.fd], [], [], 0)
            if rlist:
                data = os.read(self.fd, max_bytes)
                if not data:
                    self._alive = False
                return data
        except (OSError, IOError) as e:
            if e.errno in (errno.EIO, errno.EBADF):
                self._alive = False
            return b''
        return b''

    def write(self, data: bytes) -> None:
        """Write bytes to the PTY (keyboard input to the shell)."""
        if self._alive and self.fd >= 0:
            try:
                os.write(self.fd, data)
            except OSError:
                self._alive = False

    @property
    def alive(self) -> bool:
        """Check if child is still running."""
        if not self._alive:
            return False
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid != 0:
                self._alive = False
        except ChildProcessError:
            self._alive = False
        return self._alive

    def terminate(self) -> None:
        """Gracefully terminate the shell: SIGHUP, then SIGKILL after timeout."""
        if self.pid <= 0:
            return
        try:
            os.kill(self.pid, signal.SIGHUP)
        except ProcessLookupError:
            return

        # Wait up to 500ms for graceful exit
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            try:
                pid, _ = os.waitpid(self.pid, os.WNOHANG)
                if pid != 0:
                    return
            except ChildProcessError:
                return
            time.sleep(0.01)

        # Force kill
        try:
            os.kill(self.pid, signal.SIGKILL)
            os.waitpid(self.pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass

        # Close fd
        if self.fd >= 0:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = -1


# ---------------------------------------------------------------------------
# Keyboard mapping
# ---------------------------------------------------------------------------

def _key_to_bytes(event: pygame.event.Event) -> bytes | None:
    """Map pygame KEYDOWN to raw bytes for the PTY.

    Returns None for the quit sentinel (Ctrl+Q).
    """
    mods = event.mod

    # Ctrl+Q — quit the neural terminal (not forwarded to shell)
    if event.key == pygame.K_q and (mods & pygame.KMOD_CTRL):
        return None

    # Ctrl+<letter>
    if (mods & pygame.KMOD_CTRL) and pygame.K_a <= event.key <= pygame.K_z:
        return bytes([event.key - pygame.K_a + 1])

    # Arrow keys
    arrow_map = {
        pygame.K_UP: b'\x1b[A', pygame.K_DOWN: b'\x1b[B',
        pygame.K_RIGHT: b'\x1b[C', pygame.K_LEFT: b'\x1b[D',
    }
    if event.key in arrow_map:
        return arrow_map[event.key]

    # Function and special keys
    special = {
        pygame.K_RETURN: b'\r',
        pygame.K_BACKSPACE: b'\x7f',   # DEL (xterm convention)
        pygame.K_TAB: b'\t',
        pygame.K_ESCAPE: b'\x1b',
        pygame.K_DELETE: b'\x1b[3~',
        pygame.K_HOME: b'\x1b[H',
        pygame.K_END: b'\x1b[F',
        pygame.K_PAGEUP: b'\x1b[5~',
        pygame.K_PAGEDOWN: b'\x1b[6~',
        pygame.K_INSERT: b'\x1b[2~',
        pygame.K_F1: b'\x1bOP',
        pygame.K_F2: b'\x1bOQ',
        pygame.K_F3: b'\x1bOR',
        pygame.K_F4: b'\x1bOS',
        pygame.K_F5: b'\x1b[15~',
        pygame.K_F6: b'\x1b[17~',
        pygame.K_F7: b'\x1b[18~',
        pygame.K_F8: b'\x1b[19~',
        pygame.K_F9: b'\x1b[20~',
        pygame.K_F10: b'\x1b[21~',
        pygame.K_F11: b'\x1b[23~',
        pygame.K_F12: b'\x1b[24~',
    }
    if event.key in special:
        return special[event.key]

    # Printable characters
    if event.unicode and ord(event.unicode) >= 0x20:
        return event.unicode.encode('utf-8')

    return b''


# ---------------------------------------------------------------------------
# Display loading
# ---------------------------------------------------------------------------

def _load_display(args: argparse.Namespace) -> NeuralDisplay:
    """Load V1 or V2 neural display."""
    if args.v2:
        from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2
        path = PROJECT_ROOT / "models" / "display" / "terminal_renderer_v2.pt"
        return NeuralDisplayV2(str(path), device=args.device)
    path = PROJECT_ROOT / "models" / "display" / "terminal_renderer.pt"
    return NeuralDisplay(str(path), device=args.device)


def _render_overlay(surface: pygame.Surface, text: str, scale: int):
    """Small overlay text in top-right."""
    font = pygame.font.SysFont("monospace", max(12, 10 * scale))
    surf = font.render(text, True, (0, 255, 0), (0, 0, 0))
    rect = surf.get_rect()
    rect.topright = (surface.get_width() - 4, 4)
    surface.blit(surf, rect)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Neural PTY Terminal — real shell, every pixel neural"
    )
    parser.add_argument(
        "--shell", type=str, default=None,
        help="Shell to spawn (default: $SHELL or /bin/zsh)"
    )
    parser.add_argument("--scale", type=int, default=2, help="Window scale (default: 2)")
    parser.add_argument("--device", type=str, default=None, help="Render device")
    parser.add_argument("--v2", action="store_true", help="Use V2 renderer")
    args = parser.parse_args()

    shell = args.shell or os.environ.get("SHELL", "/bin/zsh")
    scale = max(1, args.scale)
    win_w = FRAME_W * scale
    win_h = FRAME_H * scale

    # Load neural display
    print(f"Loading neural renderer ({'V2' if args.v2 else 'V1'})...")
    display = _load_display(args)
    backend = "Metal GPU" if getattr(display, 'metal_available', False) else "PyTorch"
    print(f"Backend: {backend} | Shell: {shell}")
    print(f"Window: {win_w}x{win_h} (scale {scale}x)")
    print(f"Terminal: {TERM_ROWS}x{TERM_COLS} | Ctrl+Q to quit")

    # Spawn shell
    proc = ShellProcess(shell, TERM_ROWS, TERM_COLS)
    proc.start()
    print(f"Shell PID: {proc.pid}")

    # Initialize pygame
    pygame.init()
    pygame.display.set_caption(f"nCPU Neural Terminal - {os.path.basename(shell)}")
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()
    pygame.key.set_repeat(400, 50)

    frame_times: deque[float] = deque(maxlen=60)
    running = True

    try:
        while running:
            t0 = time.perf_counter()

            # --- Read shell output (non-blocking) ---
            data = proc.read()
            if data:
                display.write(data)

            # Check if shell exited
            if not proc.alive:
                # Render one last frame showing exit message
                display.terminal.write_str(
                    "\r\n\x1b[1;31m[shell exited]\x1b[0m\r\n"
                    "\x1b[90mClose window or press any key.\x1b[0m"
                )
                frame = display.render()
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                if scale != 1:
                    surf = pygame.transform.scale(surf, (win_w, win_h))
                screen.blit(surf, (0, 0))
                pygame.display.flip()

                # Wait for user to close
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type in (pygame.QUIT, pygame.KEYDOWN):
                            waiting = False
                            break
                    clock.tick(10)
                running = False
                break

            # --- Events ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    data = _key_to_bytes(event)
                    if data is None:  # Ctrl+Q
                        running = False
                        break
                    if data:
                        proc.write(data)

            if not running:
                break

            # --- Neural render ---
            frame = display.render()
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            if scale != 1:
                surf = pygame.transform.scale(surf, (win_w, win_h))
            screen.blit(surf, (0, 0))

            # FPS overlay
            t1 = time.perf_counter()
            frame_times.append(t1 - t0)
            avg = sum(frame_times) / len(frame_times) if frame_times else 1.0
            fps = 1.0 / avg if avg > 0 else 0.0
            _render_overlay(screen, f"{fps:.1f} FPS | PTY", scale)

            pygame.display.flip()
            clock.tick(60)

    finally:
        proc.terminate()
        pygame.quit()

    print("Neural PTY terminal closed.")


if __name__ == "__main__":
    main()
