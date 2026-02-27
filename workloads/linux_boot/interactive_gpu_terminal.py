#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    REAL INTERACTIVE GPU TERMINAL                             ║
║           Single ARM64 Binary Running Continuously on GPU                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  This is NOT a Python shell wrapper!                                         ║
║  This runs a SINGLE ARM64 binary (like busybox sh) that:                    ║
║  - Executes ALL instructions on GPU tensors                                  ║
║  - Uses read() syscall for REAL stdin input                                  ║
║  - Runs continuously until the binary exits                                  ║
║                                                                              ║
║  The shell logic is in the ARM64 binary, not Python!                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import time
import select
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from neural_kernel import NeuralARM64Kernel


def run_interactive_binary(binary_path: str, argv: list = None, max_instructions: int = 100_000_000):
    """
    Run an ARM64 binary interactively on GPU.

    The binary runs continuously. When it needs input (read syscall on stdin),
    we get input from the user and continue execution.
    """
    print(f"\n{'='*70}")
    print("  REAL GPU INTERACTIVE TERMINAL")
    print("  Running ARM64 binary continuously on GPU tensors")
    print(f"{'='*70}\n")

    # Load kernel (suppress init messages)
    print("[INIT] Creating Neural ARM64 Kernel...")

    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    kernel = NeuralARM64Kernel()
    sys.stdout = old_stdout

    print(f"[OK] Kernel ready on {kernel.cpu.device}")

    # Load binary
    with open(binary_path, 'rb') as f:
        elf_data = f.read()

    if argv is None:
        argv = [os.path.basename(binary_path)]

    print(f"[LOAD] Binary: {binary_path}")
    print(f"[LOAD] Args: {argv}")

    # Use kernel's proper run_elf_interactive method
    print()
    print("="*70)
    print("  Binary is now running on GPU. All execution on tensor cores.")
    print("  Press Ctrl+C to interrupt, Ctrl+D to exit.")
    print("="*70)
    print()

    # Run using the kernel's adaptive runner which handles everything properly
    start_time = time.perf_counter()

    try:
        # Disable syscall tracing for cleaner output
        kernel.linux_syscalls.trace_syscalls = False

        result = kernel.run_elf_adaptive(
            elf_data,
            argv=argv,
            max_instructions=max_instructions,
            ips_threshold=5000,
        )
        exit_code = result[0]

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Execution stopped by user")
        exit_code = 130

    elapsed = time.perf_counter() - start_time

    print()
    print("="*70)
    print(f"  Total instructions: {kernel.total_instructions:,}")
    print(f"  Elapsed time: {elapsed:.2f}s")
    if elapsed > 0:
        print(f"  Average IPS: {kernel.total_instructions/elapsed:,.0f}")
    print(f"  Exit code: {exit_code}")
    print("="*70)

    return exit_code


def main():
    """Main entry point."""
    # Check for binary argument
    if len(sys.argv) < 2:
        print("Usage: python3 interactive_gpu_terminal.py <binary> [args...]")
        print()
        print("Examples:")
        print("  python3 interactive_gpu_terminal.py binaries/busybox-static sh")
        print("  python3 interactive_gpu_terminal.py binaries/alpine-hello")
        return 1

    binary_path = sys.argv[1]
    if not os.path.exists(binary_path):
        # Try in binaries directory
        alt_path = Path(__file__).parent / "binaries" / binary_path
        if alt_path.exists():
            binary_path = str(alt_path)
        else:
            print(f"Error: Binary not found: {binary_path}")
            return 1

    argv = sys.argv[1:]  # Use provided arguments

    return run_interactive_binary(binary_path, argv)


if __name__ == "__main__":
    sys.exit(main())
