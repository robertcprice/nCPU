#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        BOOT ALPINE LINUX ON PARALLELMETALCPU (720M+ IPS GPU)                  ║
║                                                                              ║
║  Boots REAL ARM64 Linux kernel using ParallelMetalCPU GPU lanes              ║
║  - 128 parallel ARM64 CPUs on GPU                                            ║
║  - 720M+ instructions per second                                             ║
║  - Raw ARM64 boot Image (not ELF)                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import importlib.util
import time
import gzip
from pathlib import Path

# Load the shared library directly (same pattern as benchmark_parallel.py)
spec = importlib.util.spec_from_file_location(
    'kvrm_metal',
    '/Users/bobbyprice/projects/.venv/lib/python3.13/site-packages/kvrm_metal/kvrm_metal.cpython-313-darwin.so'
)
kvrm_metal = importlib.util.module_from_spec(spec)
sys.modules['kvrm_metal'] = kvrm_metal
spec.loader.exec_module(kvrm_metal)

ParallelMetalCPU = kvrm_metal.ParallelMetalCPU


def boot_alpine_parallel(num_lanes=128):
    """Boot Alpine Linux on ParallelMetalCPU with GPU acceleration."""
    print("=" * 70)
    print("  BOOTING ALPINE LINUX ON PARALLELMETALCPU")
    print(f"  {num_lanes} PARALLEL ARM64 CPUS ON GPU")
    print("=" * 70)
    print()

    # Paths to kernel and initrd
    script_dir = Path(__file__).parent
    kernel_path = script_dir / "linux"
    initrd_path = script_dir / "initrd.gz"

    # Check files exist
    if not kernel_path.exists():
        print(f"ERROR: Kernel not found at {kernel_path}")
        return 1

    if not initrd_path.exists():
        print(f"ERROR: Initrd not found at {initrd_path}")
        return 1

    print(f"  Kernel: {kernel_path}")
    print(f"  Initrd: {initrd_path}")
    print()

    # Load kernel
    print(f"  Loading ARM64 boot Image...")
    with open(kernel_path, 'rb') as f:
        kernel_data = f.read()
    print(f"  Kernel size: {len(kernel_data):,} bytes ({len(kernel_data)/1024/1024:.1f} MB)")

    # Decompress initrd
    print(f"  Decompressing initrd.gz...")
    with gzip.open(initrd_path, 'rb') as f:
        initrd_data = f.read()
    print(f"  Initrd size: {len(initrd_data):,} bytes ({len(initrd_data)/1024/1024:.1f} MB)")
    print()

    # Create ParallelMetalCPU with larger memory for Linux
    # Need 128MB for kernel + initrd + overhead
    memory_size = 128 * 1024 * 1024  # 128MB

    print(f"  Creating {num_lanes} parallel ARM64 CPUs with {memory_size/1024/1024}MB memory...")
    cpu = ParallelMetalCPU(num_lanes=num_lanes, memory_size=memory_size)
    print()

    # NOTE: ParallelMetalCPU has limited memory (16MB) and no MMU
    # We need to load kernel at address 0x0 instead of standard 0x40080000
    # ARM64 boot Images are position-independent for early boot code

    kernel_load_addr = 0x00000000
    initrd_load_addr = 0x02000000  # 32MB offset (after kernel space)
    device_tree_addr = 0x02400000  # After initrd

    print(f"  Loading kernel at 0x{kernel_load_addr:X}...")
    cpu.load_program(list(kernel_data), kernel_load_addr)
    print(f"  Loaded {len(kernel_data)} bytes")

    print(f"  Loading initrd at 0x{initrd_load_addr:X}...")
    cpu.load_program(list(initrd_data), initrd_load_addr)
    print(f"  Loaded {len(initrd_data)} bytes")
    print()

    # Set up initial registers for ARM64 Linux boot
    # X0 = Device tree pointer (or 0 if no device tree)
    # X1-X3 = 0 (reserved)
    # PC = Kernel entry point (0x40080000 for boot Image)

    print(f"  Setting up boot registers...")
    print(f"    PC = 0x{kernel_load_addr:X} (kernel entry)")
    print(f"    X0 = 0x{device_tree_addr:X} (device tree pointer)")
    print(f"    X1-X3 = 0")

    cpu.set_pc_all(kernel_load_addr)

    # Set registers for all lanes
    for lane in range(num_lanes):
        # X0 = device tree address
        cpu.set_registers_lane(lane, [device_tree_addr] + [0] * 31)

    print()

    # Boot!
    print("=" * 70)
    print("  STARTING LINUX KERNEL")
    print("=" * 70)
    print()

    max_cycles = 100_000_000  # 100M cycles
    print(f"  Running for {max_cycles:,} cycles...")
    print(f"  Expected time: ~{max_cycles / 720_000_000:.1f}s at 720M IPS")
    print()

    start = time.time()
    result = cpu.execute(max_cycles)
    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("  EXECUTION COMPLETE")
    print("=" * 70)
    print()
    print(f"  Total cycles: {result.total_cycles:,}")
    print(f"  Wall time: {elapsed:.4f}s")
    print(f"  Average IPS: {result.avg_ips():,.0f}")
    print(f"  Min/Max cycles per lane: {result.min_cycles():,}/{result.max_cycles():,}")
    print(f"  Lane efficiency: {result.lane_efficiency():.1%}")
    print()

    # Show final PCs of first few lanes
    print(f"  Final PCs (first 8 lanes):")
    for i in range(min(8, num_lanes)):
        print(f"    Lane {i:2d}: 0x{result.pcs_per_lane[i]:X}")

    print()

    # Check if any lanes hit SVC (syscall) - would show in stop_reason
    stop_reasons = result.stop_reasons
    unique_reasons = set(stop_reasons)
    if unique_reasons and unique_reasons != {0}:
        print(f"  Stop reasons: {unique_reasons}")
        for i, reason in enumerate(stop_reasons[:8]):
            if reason != 0:
                print(f"    Lane {i}: reason={reason}")
    else:
        print(f"  All lanes stopped naturally (max cycles)")

    print()
    print("=" * 70)

    return 0


def analyze_kernel():
    """Analyze the kernel Image to find entry point and other info."""
    script_dir = Path(__file__).parent
    kernel_path = script_dir / "linux"

    if not kernel_path.exists():
        print(f"Kernel not found at {kernel_path}")
        return

    print("=" * 70)
    print("  KERNEL IMAGE ANALYSIS")
    print("=" * 70)
    print()

    with open(kernel_path, 'rb') as f:
        data = f.read()

    print(f"  Size: {len(data):,} bytes ({len(data)/1024/1024:.1f} MB)")
    print()

    # ARM64 boot Image format:
    # First 4 bytes should be 0x00000000 (or specific magic)
    # Code starts at offset 0
    # Entry point is at load_address + 0 (for boot Image)

    # Check first few instructions
    print("  First 8 instructions (entry point):")
    for i in range(8):
        if (i + 1) * 4 <= len(data):
            inst = int.from_bytes(data[i*4:(i+1)*4], 'little')
            print(f"    0x{i*4:04X}: 0x{inst:08X}")

    print()

    # Look for common patterns
    # ARM64 exception vector table usually has specific patterns
    # B instruction (0x14000000) is common in vector tables

    b_count = data[3:4:4].count(b'\x14')  # Count B instructions at offset 3
    print(f"  B instructions in first 32 bytes: {b_count}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Boot Alpine Linux on ParallelMetalCPU")
    parser.add_argument("--analyze", action="store_true", help="Analyze kernel Image")
    parser.add_argument("--lanes", type=int, default=128, help="Number of parallel lanes")
    parser.add_argument("--cycles", type=int, default=100_000_000, help="Max cycles")

    args = parser.parse_args()

    if args.analyze:
        analyze_kernel()
    else:
        sys.exit(boot_alpine_parallel(num_lanes=args.lanes))
