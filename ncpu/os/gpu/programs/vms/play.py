#!/usr/bin/env python3
"""
VM Launcher — Compile and run virtual machines on the ARM64 Metal GPU.

Usage:
    python demos/vms/play.py bf              # Interactive Brainfuck REPL
    echo "program" | python demos/vms/play.py bf   # Pipe BF program
    python demos/vms/play.py forth           # Interactive Forth REPL
    python demos/vms/play.py chip8           # CHIP-8 with built-in demo ROM
"""

import sys
import os
import tempfile
from pathlib import Path

VMS_DIR = Path(__file__).parent
GPU_OS_DIR = VMS_DIR.parent.parent
PROJECT_ROOT = GPU_OS_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.runner import compile_c, run, make_syscall_handler
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2


VMS = {
    "bf": "brainfuck.c",
    "brainfuck": "brainfuck.c",
    "forth": "forth.c",
    "chip8": "chip8.c",
}


def run_vm(vm_name: str):
    canonical = vm_name.lower()
    if canonical not in VMS:
        print(f"Unknown VM: {vm_name}")
        print(f"Available: bf, forth, chip8")
        sys.exit(1)

    c_file = VMS_DIR / VMS[canonical]
    if not c_file.exists():
        print(f"VM source not found: {c_file}")
        sys.exit(1)

    # Compile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name

    print(f"Compiling {canonical}...")
    if not compile_c(str(c_file), bin_path):
        print("Compilation failed!")
        sys.exit(1)

    binary = Path(bin_path).read_bytes()
    print(f"Binary: {len(binary):,} bytes")

    # Set up GPU
    cpu = MLXKernelCPUv2()
    cpu.load_program(binary, address=0x10000)
    cpu.set_pc(0x10000)

    # For BF: check if stdin has piped data
    piped_input = None
    if not sys.stdin.isatty():
        piped_input = sys.stdin.buffer.read()
        piped_offset = [0]

    def on_read(fd, max_len):
        if fd == 0 and piped_input is not None:
            start = piped_offset[0]
            end = min(start + max_len, len(piped_input))
            if start >= len(piped_input):
                return b""
            data = piped_input[start:end]
            piped_offset[0] = end
            return data
        return None  # Fall through to default stdin

    # For CHIP-8: use getchar with raw mode
    needs_raw = canonical == "chip8"

    old_settings = None
    if needs_raw and sys.stdin.isatty():
        import tty, termios
        old_settings = termios.tcgetattr(sys.stdin.fileno())

    def on_getchar():
        import select
        if sys.stdin.isatty():
            rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
            if rlist:
                return ord(sys.stdin.read(1))
        return -1

    handler = make_syscall_handler(
        on_read=on_read if piped_input is not None else None,
        on_getchar=on_getchar,
    )

    try:
        if old_settings:
            import tty, termios
            tty.setcbreak(sys.stdin.fileno())

        print(f"Starting {canonical}...")
        print("=" * 60)
        results = run(cpu, handler, max_cycles=500_000_000, quiet=True)
    finally:
        if old_settings:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
        if os.path.exists(bin_path):
            os.unlink(bin_path)

    print()
    print("=" * 60)
    print(f"VM: {canonical}")
    print(f"Cycles: {results['total_cycles']:,}")
    print(f"Elapsed: {results['elapsed']:.3f}s")
    print(f"IPS: {results['ips']:,.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demos/vms/play.py <vm>")
        print("Available: bf, forth, chip8")
        sys.exit(1)

    run_vm(sys.argv[1])
