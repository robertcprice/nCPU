#!/usr/bin/env python3
"""
Conway's Game of Life — Compiled C running on ARM64 Metal GPU.

Compiles arm64_game_of_life.c with aarch64-elf-gcc, loads the raw binary
onto the MLX Metal GPU kernel, and runs with Python-mediated syscall I/O.

The C program runs entirely on GPU. Python only handles SVC traps for
displaying the grid (fd=3 signals).

Usage:
    python demos/arm64_game_of_life.py
"""

import sys
import os
import struct
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.runner import compile_c, run, make_syscall_handler
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2

ROWS = 20
COLS = 20


def render_grid(grid_bytes: bytes, generation: int):
    """Render a 20x20 grid as ASCII art."""
    # ANSI: move cursor up to overwrite previous frame (except first)
    if generation > 0:
        print(f"\033[{ROWS + 2}A", end="")

    print(f"╔{'══' * COLS}╗  Generation {generation:3d}")
    for r in range(ROWS):
        print("║", end="")
        for c in range(COLS):
            cell = grid_bytes[r * COLS + c]
            print("██" if cell else "  ", end="")
        print("║")
    print(f"╚{'══' * COLS}╝")


def main():
    print("=" * 60)
    print("  Conway's Game of Life")
    print("  Compiled C on ARM64 Metal GPU")
    print("=" * 60)
    print()

    c_file = PROJECT_ROOT / "ncpu" / "os" / "gpu" / "src" / "arm64_game_of_life.c"

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name

    try:
        if not compile_c(str(c_file), bin_path):
            sys.exit(1)

        binary = Path(bin_path).read_bytes()
        cpu = MLXKernelCPUv2()
        cpu.load_program(binary, address=0x10000)
        cpu.set_pc(0x10000)

        print(f"Grid: {ROWS}x{COLS} toroidal")
        print("Patterns: glider + blinker + block")
        print()

        # Custom write handler: fd=3 → grid display
        def on_write(fd, data):
            if fd == 3 and len(data) >= 16:
                # Data is [grid_ptr (8 bytes), generation (8 bytes)]
                grid_ptr, generation = struct.unpack("<qq", data[:16])
                # Read grid from GPU memory
                grid_data = cpu.read_memory(grid_ptr, ROWS * COLS)
                render_grid(grid_data, generation)
                time.sleep(0.15)  # Animation delay
                return True  # Handled
            return False  # Use default handler

        handler = make_syscall_handler(on_write=on_write)
        results = run(cpu, handler, max_cycles=50_000_000, quiet=True)

        print()
        print(f"Total cycles: {results['total_cycles']:,}")
        print(f"Elapsed: {results['elapsed']:.3f}s")
        print(f"IPS: {results['ips']:,.0f}")

    finally:
        if os.path.exists(bin_path):
            os.unlink(bin_path)


if __name__ == "__main__":
    main()
