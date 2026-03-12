#!/usr/bin/env python3
"""
Interactive Shell — Compiled C running on ARM64 Metal GPU.

Compiles arm64_shell.c with aarch64-elf-gcc, loads onto the Metal GPU kernel,
and runs with Python-mediated syscall I/O. The shell reads from stdin (via
SYS_READ trap) and writes to stdout (via SYS_WRITE trap).

Usage:
    python ncpu/os/gpu/shell.py
"""

import sys
import os
import tempfile
from pathlib import Path

GPU_OS_DIR = Path(__file__).parent
PROJECT_ROOT = GPU_OS_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.runner import compile_c, run, make_syscall_handler
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2


def main():
    print("=" * 60)
    print("  ARM64 Interactive Shell on Metal GPU")
    print("  C → GCC → Binary → Metal Compute Shader → Python I/O")
    print("=" * 60)
    print()

    c_file = GPU_OS_DIR / "src" / "arm64_shell.c"

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name

    try:
        if not compile_c(str(c_file), bin_path):
            sys.exit(1)

        binary = Path(bin_path).read_bytes()
        cpu = MLXKernelCPUv2()
        cpu.load_program(binary, address=0x10000)
        cpu.set_pc(0x10000)

        handler = make_syscall_handler()
        results = run(cpu, handler, max_cycles=100_000_000, quiet=True)

        print()
        print(f"Total cycles: {results['total_cycles']:,}")
        print(f"Elapsed: {results['elapsed']:.3f}s")
        print(f"IPS: {results['ips']:,.0f}")

    finally:
        if os.path.exists(bin_path):
            os.unlink(bin_path)


if __name__ == "__main__":
    main()
