#!/usr/bin/env python3
"""Live neural OS with real-time pygame display.

Boots the neural-enhanced GPU UNIX OS and renders every output byte through
the V2 neural display (390K params) in a live pygame window. Keystrokes in
the pygame window are forwarded to the shell running on Metal GPU.

Architecture:
  - OS thread: Metal GPU executes ARM64, Python handles syscalls. SYS_READ
    on fd 0 blocks on a queue until the user types in the pygame window.
  - Main thread: pygame event loop, neural display rendering at ~10 FPS,
    keyboard input forwarded to OS via queue.
  - Shared state: NeuralDisplayV2.terminal (TerminalState) is written by
    the OS thread (via SYS_WRITE -> neural_display.write()) and read by
    the main thread (via display.render()). A threading.Lock serializes
    access.

Usage:
    python ncpu/os/gpu/neural_live.py
    python ncpu/os/gpu/neural_live.py --scale 2
    python ncpu/os/gpu/neural_live.py --device cpu
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress Metal V2 kernel warning (not compiled into .so yet, PyTorch fallback expected)
logging.getLogger("ncpu.neural.metal_inference").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Terminal geometry (from V1 renderer)
# ---------------------------------------------------------------------------

from ncpu.neural.neural_terminal_renderer import FRAME_H, FRAME_W


# ---------------------------------------------------------------------------
# Pygame key mapping (reuse from neural_display_live.py)
# ---------------------------------------------------------------------------

def _pygame_key_to_bytes(event) -> Optional[bytes]:
    """Convert a pygame KEYDOWN event to terminal bytes.

    Returns None for events that should not be forwarded (e.g. ESC to quit).
    Returns b'' for unrecognised keys.
    """
    import pygame

    key = event.key
    mods = event.mod

    # Ctrl+C -> SIGINT byte
    if key == pygame.K_c and (mods & pygame.KMOD_CTRL):
        return b"\x03"
    # Ctrl+D -> EOF
    if key == pygame.K_d and (mods & pygame.KMOD_CTRL):
        return b"\x04"
    # Ctrl+L -> form feed (clear)
    if key == pygame.K_l and (mods & pygame.KMOD_CTRL):
        return b"\x0c"

    # Arrow keys -> VT100 escape sequences
    if key == pygame.K_UP:
        return b"\033[A"
    if key == pygame.K_DOWN:
        return b"\033[B"
    if key == pygame.K_RIGHT:
        return b"\033[C"
    if key == pygame.K_LEFT:
        return b"\033[D"

    # Special keys
    if key == pygame.K_RETURN:
        return b"\n"
    if key == pygame.K_BACKSPACE:
        return b"\x7f"
    if key == pygame.K_TAB:
        return b"\t"
    if key == pygame.K_DELETE:
        return b"\033[3~"

    # Escape quits the app (handled by caller)
    if key == pygame.K_ESCAPE:
        return None  # sentinel: quit

    # Printable character
    if event.unicode and ord(event.unicode) >= 0x20:
        return event.unicode.encode("utf-8")

    return b""


# ---------------------------------------------------------------------------
# Live Neural OS
# ---------------------------------------------------------------------------

class LiveNeuralOS:
    """Combines GPU OS execution with live neural display rendering.

    The GPU OS (compiled C shell running ARM64 on Metal) executes in a daemon
    thread. The main thread runs the pygame display loop, rendering the shared
    TerminalState through the V2 neural display at interactive frame rates.
    """

    def __init__(self, scale: int = 2, device: str = "mps"):
        self.scale = scale
        self.device = device
        self.running = True
        self.os_ready = threading.Event()  # set once OS thread has booted
        self.input_queue: queue.Queue[bytes] = queue.Queue()
        self.needs_render = True

        # Lock protects TerminalState reads (render) vs writes (OS output)
        self._term_lock = threading.Lock()

        # Neural display (shared between threads)
        from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2
        self.display = NeuralDisplayV2(device=device)

        # Force PyTorch path for rendering -- the Metal V2 native kernel is not
        # yet optimized (falls back to a slow Python wrapper), while the PyTorch
        # MPS path achieves 15-20 FPS which exceeds our 10 FPS target.
        if self.display._use_metal:
            self.display._use_metal = False

        # Track FPS for title bar
        self._frame_count = 0
        self._fps_timer = 0.0
        self._display_fps = 0.0
        self._os_ips = 0

        # Neural tab completion: command suggestor learns from session and
        # suggests completions when the user presses TAB
        from ncpu.os.gpu.neural_demo import (
            NeuralCommandSuggestor, NeuralErrorRecovery,
        )
        self.command_suggestor = NeuralCommandSuggestor()
        self.error_recovery = NeuralErrorRecovery()
        self._input_buffer = ""  # tracks current line for TAB completion

    # ── OS thread ─────────────────────────────────────────────────────────

    def _os_thread(self):
        """Boot and run the GPU OS in a background thread."""
        from ncpu.os.gpu.runner import (
            compile_c,
            make_syscall_handler,
            StopReasonV2,
            HEAP_BASE,
        )
        from ncpu.os.gpu.filesystem import GPUFilesystem
        from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2, StopReasonV2

        GPU_OS_DIR = Path(__file__).parent

        # ── Bootstrap filesystem ──────────────────────────────────────────
        fs = GPUFilesystem()
        for d in ["/home/user", "/var/log", "/usr/lib", "/tmp"]:
            fs.mkdir(d)

        fs.write_file(
            "/etc/motd",
            "Welcome to nCPU Neural Computer v3.1\n"
            "Live neural display -- every pixel is neural (390K params)\n"
            "Type 'help' for commands.\n",
        )
        fs.write_file("/etc/hostname", "ncpu\n")
        fs.write_file(
            "/home/user/hello.c",
            '#include "arm64_libc.h"\n'
            "\n"
            "int main(void) {\n"
            '    printf("Hello from GPU-compiled C!\\n");\n'
            '    printf("Running on Metal silicon with neural display.\\n");\n'
            "    return 0;\n"
            "}\n",
        )
        fs.write_file(
            "/home/user/fib.c",
            '#include "arm64_libc.h"\n'
            "\n"
            "int main(void) {\n"
            '    printf("Fibonacci sequence:\\n");\n'
            "    long a = 0, b = 1;\n"
            "    for (int i = 0; i < 20; i++) {\n"
            '        printf("  fib(%d) = %ld\\n", i, a);\n'
            "        long tmp = a + b;\n"
            "        a = b;\n"
            "        b = tmp;\n"
            "    }\n"
            "    return 0;\n"
            "}\n",
        )
        fs.chdir("/home/user")

        # ── Compile shell ─────────────────────────────────────────────────
        c_file = GPU_OS_DIR / "src" / "arm64_unix_shell.c"
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            bin_path = f.name

        if not compile_c(str(c_file), bin_path, quiet=True):
            print("[live] FATAL: Shell compilation failed")
            self.running = False
            self.os_ready.set()
            return

        binary = Path(bin_path).read_bytes()

        # ── Create Metal CPU and load shell ───────────────────────────────
        cpu = MLXKernelCPUv2()
        cpu.load_program(binary, address=0x10000)
        cpu.set_pc(0x10000)

        # ── on_read: block on input queue for fd 0 ───────────────────────
        def on_read(fd: int, max_len: int) -> Optional[bytes]:
            if fd != 0:
                return None
            # Block until input arrives (with periodic timeout to check self.running)
            while self.running:
                try:
                    data = self.input_queue.get(timeout=0.1)
                    # Feed completed commands to the suggestor for learning
                    try:
                        text = data.decode('ascii', errors='replace')
                        if '\n' in text:
                            cmd = text.strip()
                            if cmd:
                                self.command_suggestor.observe(cmd)
                                self.error_recovery.set_current_command(cmd)
                    except Exception:
                        pass
                    return data[:max_len]
                except queue.Empty:
                    continue
            # Shutting down -- return EOF
            return b""

        # ── on_exec: reload binary ────────────────────────────────────────
        def on_exec(bin_path_str: str) -> bool:
            resolved = fs.resolve_path(bin_path_str)
            binary_data = fs.read_file(resolved)
            if binary_data:
                cpu.load_program(binary_data, address=0x10000)
                cpu.set_pc(0x10000)
                return True
            return False

        # ── Wrap neural_display.write with lock ───────────────────────────
        original_write = self.display.write

        def locked_write(data: bytes):
            with self._term_lock:
                original_write(data)
            self.needs_render = True

        self.display.write = locked_write  # type: ignore[assignment]

        # ── on_write: error recovery scanning ─────────────────────────────
        def on_write(fd, data):
            """Scan SYS_WRITE for error patterns and emit recovery suggestions."""
            if fd not in (1, 2):
                return False
            try:
                text = data.decode('ascii', errors='replace') if isinstance(data, (bytes, bytearray)) else str(data)
                suggestion = self.error_recovery.analyze_output(text)
                if suggestion:
                    clean = suggestion.replace("\033[93m", "").replace("\033[0m", "")
                    with self._term_lock:
                        original_write((clean + "\n").encode('utf-8', errors='replace'))
                    self.needs_render = True
            except Exception:
                pass
            return False  # Never suppress the original write

        # ── Syscall handler ───────────────────────────────────────────────
        handler = make_syscall_handler(
            filesystem=fs,
            on_read=on_read,
            on_write=on_write,
            on_exec=on_exec,
            neural_display=self.display,
        )

        # ── Signal ready ──────────────────────────────────────────────────
        self.os_ready.set()

        # ── GPU execution loop (inline from runner.run()) ─────────────────
        batch_size = 100_000
        max_cycles = 500_000_000
        total_cycles = 0
        t0 = time.perf_counter()

        # Initialize GPU-side SVC write buffer
        if cpu.memory_size > cpu.SVC_BUF_BASE + 0x10000:
            cpu.init_svc_buffer()

        def _drain_gpu_writes():
            """Drain GPU-buffered SYS_WRITE entries."""
            if cpu.memory_size <= cpu.SVC_BUF_BASE:
                return
            for fd, data in cpu.drain_svc_buffer():
                if fd in (1, 2):
                    with self._term_lock:
                        original_write(
                            bytes(data) if not isinstance(data, bytes) else data
                        )
                    self.needs_render = True

        while self.running and total_cycles < max_cycles:
            result = cpu.execute(max_cycles=batch_size)
            total_cycles += result.cycles

            # Update IPS estimate periodically
            elapsed = time.perf_counter() - t0
            if elapsed > 0:
                self._os_ips = int(total_cycles / elapsed)

            _drain_gpu_writes()

            if result.stop_reason == StopReasonV2.HALT:
                break
            elif result.stop_reason == StopReasonV2.SYSCALL:
                ret = handler(cpu)
                if ret is False:
                    break
                elif ret == "exec":
                    continue
                cpu.set_pc(cpu.pc + 4)
            elif result.stop_reason == StopReasonV2.MAX_CYCLES:
                continue

        self.running = False

    # ── Main pygame loop ──────────────────────────────────────────────────

    def run_live(self):
        """Main loop: pygame display + keyboard input."""
        try:
            import pygame
        except ImportError:
            print()
            print("  [ERROR] pygame is required for live neural display.")
            print("  Install with: pip install pygame")
            print()
            sys.exit(1)

        pygame.init()
        win_w = FRAME_W * self.scale
        win_h = FRAME_H * self.scale
        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("nCPU Neural Computer -- Booting...")
        clock = pygame.time.Clock()

        # Enable key repeat for held keys (250ms delay, 30ms interval)
        pygame.key.set_repeat(250, 30)

        # Start OS thread
        os_thread = threading.Thread(target=self._os_thread, daemon=True)
        os_thread.start()

        # Wait for OS to boot (with progress display)
        boot_start = time.perf_counter()
        while not self.os_ready.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    return

            elapsed = time.perf_counter() - boot_start
            pygame.display.set_caption(
                f"nCPU Neural Computer -- Booting... ({elapsed:.1f}s)"
            )
            clock.tick(10)

        if not self.running:
            pygame.quit()
            return

        pygame.display.set_caption("nCPU Neural Computer -- Live Neural Display")

        # Initial render
        last_render = 0.0
        render_interval = 1.0 / 15  # 15 FPS target for rendering
        self._fps_timer = time.perf_counter()
        self._frame_count = 0

        while self.running:
            # ── Handle pygame events ──────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    # TAB: neural tab completion
                    if event.key == pygame.K_TAB:
                        suggestion = self.command_suggestor.suggest()
                        if suggestion:
                            # Insert the suggestion as if the user typed it
                            completion = (suggestion + " ").encode('utf-8')
                            self.input_queue.put(completion)
                            self.needs_render = True
                        else:
                            self.input_queue.put(b'\t')
                            self.needs_render = True
                        continue

                    data = _pygame_key_to_bytes(event)
                    if data is None:
                        # ESC -> quit
                        self.running = False
                        break
                    if data:
                        # Track input buffer for neuralstats interception
                        try:
                            text = data.decode('ascii', errors='replace')
                            if text == '\n':
                                cmd = self._input_buffer.strip()
                                if cmd == 'neuralstats':
                                    # Intercept: print stats through neural display
                                    from ncpu.os.gpu.neural_demo import (
                                        format_neural_stats, NeuralModelStatus,
                                    )
                                    _status = NeuralModelStatus()
                                    stats_text = format_neural_stats(
                                        status=_status,
                                        error_recovery=self.error_recovery,
                                    )
                                    with self._term_lock:
                                        self.display.write(stats_text.encode('utf-8', errors='replace'))
                                    self.needs_render = True
                                    # Send harmless echo to shell
                                    self.input_queue.put(b"echo Neural stats displayed.\n")
                                    self._input_buffer = ""
                                    continue
                                self._input_buffer = ""
                            elif text == '\x7f':  # backspace
                                self._input_buffer = self._input_buffer[:-1]
                            else:
                                self._input_buffer += text
                        except Exception:
                            pass
                        self.input_queue.put(data)
                        self.needs_render = True

            if not self.running:
                break

            # ── Render neural display ─────────────────────────────────────
            now = time.perf_counter()
            if self.needs_render and (now - last_render) >= render_interval:
                with self._term_lock:
                    frame = self.display.render()  # (384, 640, 3) uint8

                # Convert numpy (H, W, 3) -> pygame surface (W, H, 3)
                surface = pygame.surfarray.make_surface(
                    np.transpose(frame, (1, 0, 2))
                )
                if self.scale != 1:
                    surface = pygame.transform.scale(surface, (win_w, win_h))
                screen.blit(surface, (0, 0))
                pygame.display.flip()
                last_render = now
                self.needs_render = False

                # FPS tracking
                self._frame_count += 1
                if now - self._fps_timer >= 1.0:
                    self._display_fps = self._frame_count / (now - self._fps_timer)
                    self._frame_count = 0
                    self._fps_timer = now
                    ips_str = f"{self._os_ips:,}" if self._os_ips > 0 else "..."
                    pygame.display.set_caption(
                        f"nCPU Neural Computer -- "
                        f"{self._display_fps:.1f} FPS | "
                        f"{ips_str} IPS | "
                        f"ESC to quit"
                    )

            clock.tick(30)  # 30 Hz event polling

        pygame.quit()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="nCPU Live Neural Display -- GPU OS with real-time neural rendering"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Window scale factor (default: 2, so 1280x768)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: cpu, mps, cuda (default: auto-detect)",
    )
    args = parser.parse_args()

    # Auto-detect device
    device = args.device
    if device is None:
        import torch

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print()
    print("=" * 62)
    print("  nCPU Neural Computer -- Live Display Mode")
    print("=" * 62)
    print()
    print(f"  Device:  {device}")
    print(f"  Scale:   {args.scale}x ({FRAME_W * args.scale}x{FRAME_H * args.scale})")
    print(f"  Display: Neural V2 (390K params, 1024 chars, 256 colors)")
    print(f"  Shell:   ARM64 compiled C on Metal GPU")
    print()
    print("  Every pixel on screen is produced by neural network forward passes.")
    print("  Press ESC to quit.")
    print()

    live = LiveNeuralOS(scale=args.scale, device=device)
    live.run_live()

    print()
    print("  Session ended.")
    print()


if __name__ == "__main__":
    main()
