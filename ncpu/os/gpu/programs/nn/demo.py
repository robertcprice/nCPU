#!/usr/bin/env python3
"""
MNIST Neural Network Demo — Run a neural network on the ARM64 Metal GPU.

The meta demo: a neural network running as compiled C on a Metal compute shader
that was designed to be a neural CPU. Neural all the way down.

Usage:
    python demos/nn/demo.py
"""

import sys
import os
import tempfile
from pathlib import Path

NN_DIR = Path(__file__).parent
GPU_OS_DIR = NN_DIR.parent.parent
PROJECT_ROOT = GPU_OS_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.runner import compile_c, run, make_syscall_handler
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2


def main():
    banner = r"""
 ███    ██ ███    ██     ██████  ███    ██      ██████  ██████  ██    ██
 ████   ██ ████   ██    ██    ██ ████   ██     ██       ██   ██ ██    ██
 ██ ██  ██ ██ ██  ██    ██    ██ ██ ██  ██     ██   ███ ██████  ██    ██
 ██  ██ ██ ██  ██ ██    ██    ██ ██  ██ ██     ██    ██ ██      ██    ██
 ██   ████ ██   ████     ██████  ██   ████      ██████  ██       ██████

 Neural Network on GPU — The Recursive Masterpiece
 ──────────────────────────────────────────────────
 MNIST classifier in C → ARM64 → Metal compute shader
 On a CPU built from neural networks. Neural all the way down.
"""
    print(banner)

    # Compile
    c_file = NN_DIR / "mnist.c"
    if not c_file.exists():
        print(f"FATAL: {c_file} not found")
        sys.exit(1)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name

    print("Compiling MNIST neural network...")
    if not compile_c(str(c_file), bin_path):
        print("Compilation failed!")
        sys.exit(1)

    binary = Path(bin_path).read_bytes()
    print(f"Binary: {len(binary):,} bytes")

    # Run on GPU
    cpu = MLXKernelCPUv2()
    cpu.load_program(binary, address=0x10000)
    cpu.set_pc(0x10000)

    handler = make_syscall_handler()

    print("Running on Metal GPU...")
    print("=" * 60)

    results = run(cpu, handler, max_cycles=500_000_000, quiet=True)

    print()
    print("=" * 60)
    print(f"Cycles: {results['total_cycles']:,}")
    print(f"Elapsed: {results['elapsed']:.3f}s")
    print(f"IPS: {results['ips']:,.0f}")

    if os.path.exists(bin_path):
        os.unlink(bin_path)


if __name__ == "__main__":
    main()
