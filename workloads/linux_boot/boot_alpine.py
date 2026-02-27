#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BOOT ALPINE LINUX ON NEURAL CPU                           ║
║                                                                              ║
║  This boots REAL busybox sh directly - NO Python shell wrapper!             ║
║  The shell runs entirely through the neural ARM64 CPU tensor operations.    ║
║                                                                              ║
║  - All instructions decoded by neural tensor lookup tables                  ║
║  - All registers/memory are GPU tensors                                      ║
║  - stdin/stdout go through ARM64 syscalls → Python only for actual I/O      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path


def boot_alpine():
    """Boot directly into busybox sh - no Python wrapper."""
    print("=" * 70)
    print("  BOOTING ALPINE LINUX ON NEURAL ARM64 CPU")
    print("=" * 70)
    print()
    print("  Loading neural CPU and busybox...")
    print()

    # Import after banner (so user sees something immediately)
    from neural_kernel import NeuralARM64Kernel

    # Create kernel
    kernel = NeuralARM64Kernel()

    # Prefer the tensor-backed binary already loaded by the kernel
    candidates = [
        "/bin/busybox-static",
        "/bin/busybox-arm64-static",
        "/bin/busybox",
        "/bin/busybox-arm64",
    ]
    busybox_path = next((p for p in candidates if p in kernel.files), None)

    if busybox_path is None:
        # Fallback to disk if not preloaded
        binaries_dir = Path(__file__).parent / "binaries"
        disk_path = binaries_dir / "busybox-static"
        if not disk_path.exists():
            print(f"ERROR: {disk_path} not found!")
            print("Need a statically-linked busybox ARM64 binary.")
            return 1
        with open(disk_path, 'rb') as f:
            busybox_binary = f.read()
    else:
        busybox_binary = kernel._file_to_bytes(busybox_path)

    print(f"  Busybox binary: {len(busybox_binary):,} bytes")
    print()
    print("=" * 70)
    print("  ENTERING BUSYBOX SH - Real ARM64 shell on Neural CPU!")
    print("  Type 'exit' to quit, all commands run through neural tensors")
    print("=" * 70)
    print()

    # Run busybox sh - this will handle its own input/output through syscalls
    # argv[0] = "sh" tells busybox to run as a shell
    try:
        exit_code, elapsed = kernel.run_elf(busybox_binary, ["sh"])
        print()
        print("=" * 70)
        print(f"  Shell exited with code {exit_code}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Instructions: {kernel.total_instructions:,}")
        if elapsed > 0:
            print(f"  Average IPS: {kernel.total_instructions/elapsed:,.0f}")
        print("=" * 70)
        return exit_code
    except KeyboardInterrupt:
        print("\n  Interrupted!")
        return 130
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(boot_alpine())
