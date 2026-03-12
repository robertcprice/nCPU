#!/usr/bin/env python3
"""
Game Launcher — Compile and run games on the ARM64 Metal GPU.

Usage:
    python demos/games/play.py tetris
    python demos/games/play.py snake
    python demos/games/play.py roguelike
    python demos/games/play.py adventure
"""

import sys
import os
import tty
import termios
import tempfile
from pathlib import Path

GAMES_DIR = Path(__file__).parent
GPU_OS_DIR = GAMES_DIR.parent.parent
PROJECT_ROOT = GPU_OS_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.runner import compile_c, run, make_syscall_handler
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2


GAMES = {
    "tetris": "tetris.c",
    "snake": "snake.c",
    "roguelike": "roguelike.c",
    "adventure": "adventure.c",
}


def play_game(game_name: str):
    if game_name not in GAMES:
        print(f"Unknown game: {game_name}")
        print(f"Available: {', '.join(GAMES.keys())}")
        sys.exit(1)

    c_file = GAMES_DIR / GAMES[game_name]
    if not c_file.exists():
        print(f"Game source not found: {c_file}")
        sys.exit(1)

    # Compile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name

    print(f"Compiling {game_name}...")
    if not compile_c(str(c_file), bin_path):
        print("Compilation failed!")
        sys.exit(1)

    binary = Path(bin_path).read_bytes()
    print(f"Binary: {len(binary):,} bytes")

    # Set up GPU
    cpu = MLXKernelCPUv2()
    cpu.load_program(binary, address=0x10000)
    cpu.set_pc(0x10000)

    # Determine if game needs raw input
    needs_raw = game_name in ("tetris", "snake", "roguelike")

    # Set up raw terminal for interactive games
    old_settings = None
    if needs_raw and sys.stdin.isatty():
        old_settings = termios.tcgetattr(sys.stdin.fileno())

    def on_getchar():
        """Non-blocking single character read."""
        import select
        if sys.stdin.isatty():
            rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
            if rlist:
                return ord(sys.stdin.read(1))
        return -1  # No input available

    handler = make_syscall_handler(
        on_getchar=on_getchar,
        raw_input_mode=False,
    )

    try:
        if old_settings:
            tty.setcbreak(sys.stdin.fileno())

        print(f"Starting {game_name}...")
        print("=" * 60)
        results = run(cpu, handler, max_cycles=500_000_000, quiet=True)
    finally:
        # Restore terminal
        if old_settings:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
        # Clean up
        if os.path.exists(bin_path):
            os.unlink(bin_path)

    print()
    print("=" * 60)
    print(f"Game: {game_name}")
    print(f"Cycles: {results['total_cycles']:,}")
    print(f"Elapsed: {results['elapsed']:.3f}s")
    print(f"IPS: {results['ips']:,.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demos/games/play.py <game>")
        print(f"Available: {', '.join(GAMES.keys())}")
        sys.exit(1)

    play_game(sys.argv[1])
