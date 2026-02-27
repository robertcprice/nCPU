#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    NEURAL OS - INTERACTIVE RUNNER                                ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  Run the Neural OS with these commands:                                          ║
║                                                                                  ║
║    python run_neural_os.py              - Boot and show framebuffer              ║
║    python run_neural_os.py doom         - Run DOOM raycaster                     ║
║    python run_neural_os.py ls           - List files                             ║
║    python run_neural_os.py help         - Show help                              ║
║    python run_neural_os.py benchmark    - Run speed benchmark                    ║
║                                                                                  ║
║  PERFORMANCE:                                                                    ║
║  • Vectorized loops: 50,000,000+ IPS (executed as single tensor ops!)            ║
║  • Non-vectorized: ~2,000-5,000 IPS (neural extraction overhead)                 ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_os_final import NeuralOS, FB_WIDTH, FB_HEIGHT
from neural_cpu import device
import torch


def print_framebuffer(fb_str: str, title: str = ""):
    """Print framebuffer with nice borders."""
    lines = fb_str.split('\n')
    width = FB_WIDTH + 2

    print()
    if title:
        print(f"┌─ {title} " + "─" * (width - len(title) - 4) + "┐")
    else:
        print("┌" + "─" * width + "┐")

    for line in lines:
        # Pad to FB_WIDTH
        padded = line.ljust(FB_WIDTH)[:FB_WIDTH]
        print(f"│{padded}│")

    print("└" + "─" * width + "┘")


def run_benchmark():
    """Run speed benchmark to show max IPS."""
    print("\n" + "=" * 70)
    print("   NEURAL CPU SPEED BENCHMARK")
    print("=" * 70)

    from neural_cpu import NeuralCPU

    cpu = NeuralCPU()

    # Test 1: Vectorized count-up loop (100k iterations)
    print("\n[1] VECTORIZED LOOP (100,000 iterations):")
    code = bytearray()
    code.extend((0xD2800000).to_bytes(4, 'little'))  # MOVZ X0, #0
    code.extend((0xD290D401).to_bytes(4, 'little'))  # MOVZ X1, #0x86A0
    code.extend((0xF2A00021).to_bytes(4, 'little'))  # MOVK X1, #0x1, LSL#16 = 100000
    code.extend((0x91000400).to_bytes(4, 'little'))  # ADD X0, X0, #1
    code.extend((0xEB01001F).to_bytes(4, 'little'))  # CMP X0, X1
    code.extend((0x54FFFFCB).to_bytes(4, 'little'))  # B.LT -2
    code.extend((0x00000000).to_bytes(4, 'little'))  # halt

    cpu.load_binary(bytes(code), 0)

    # Warm up
    cpu.run(100)
    cpu.pc = torch.tensor(0, dtype=torch.int64, device=device)
    cpu.regs[0] = 0
    cpu.regs[1] = 0
    cpu.halted = False
    cpu.inst_count = torch.tensor(0, dtype=torch.int64, device=device)
    cpu.loops_vectorized = 0

    start = time.perf_counter()
    executed, _ = cpu.run(500000)
    elapsed = time.perf_counter() - start

    print(f"    Instructions: {executed:,}")
    print(f"    Time: {elapsed:.4f}s")
    print(f"    IPS: {executed/elapsed:,.0f}")
    print(f"    Loops vectorized: {cpu.loops_vectorized}")

    # Test 2: Memory fill (framebuffer clear)
    print("\n[2] VECTORIZED MEMORY FILL (2000 bytes):")
    cpu2 = NeuralCPU()

    FB_BASE = 0x40000
    code2 = bytearray()
    code2.extend((0xD2880000).to_bytes(4, 'little'))  # MOVZ X0, #FB_BASE
    code2.extend((0xD283E801).to_bytes(4, 'little'))  # MOVZ X1, #2000
    code2.extend((0xD2800402).to_bytes(4, 'little'))  # MOVZ X2, #' '
    code2.extend((0x39000002).to_bytes(4, 'little'))  # STRB W2, [X0]
    code2.extend((0x91000400).to_bytes(4, 'little'))  # ADD X0, X0, #1
    code2.extend((0xD1000421).to_bytes(4, 'little'))  # SUB X1, X1, #1
    code2.extend((0xB5FFFFA1).to_bytes(4, 'little'))  # CBNZ X1, -3
    code2.extend((0x00000000).to_bytes(4, 'little'))  # halt

    cpu2.load_binary(bytes(code2), 0)

    start = time.perf_counter()
    executed2, _ = cpu2.run(50000)
    elapsed2 = time.perf_counter() - start

    print(f"    Instructions: {executed2:,}")
    print(f"    Time: {elapsed2:.4f}s")
    print(f"    IPS: {executed2/elapsed2:,.0f}")
    print(f"    Loops vectorized: {cpu2.loops_vectorized}")

    print("\n" + "=" * 70)
    total = executed + executed2
    total_time = elapsed + elapsed2
    print(f"   TOTAL: {total:,} instructions in {total_time:.4f}s")
    print(f"   AVERAGE IPS: {total/total_time:,.0f}")
    print("=" * 70)


def run_doom():
    """Run DOOM on the Neural OS."""
    print("\n" + "=" * 70)
    print("   NEURAL OS - DOOM RAYCASTER")
    print("=" * 70)

    nos = NeuralOS()

    # Boot
    print("\nBooting...")
    nos.run(100000)  # Quick boot

    # Run DOOM
    print("Running DOOM...")
    nos.set_input("doom")

    start = time.perf_counter()
    executed, elapsed = nos.run(300000)
    total_time = time.perf_counter() - start

    print(f"\nDOOM rendered in {total_time:.2f}s")
    print(f"Instructions: {executed:,}")
    print(f"Loops vectorized: {nos.cpu.loops_vectorized}")

    print_framebuffer(nos.get_framebuffer(), "DOOM")

    nos.cpu.print_stats()


def run_command(cmd: str):
    """Run a specific command on the Neural OS."""
    print("\n" + "=" * 70)
    print(f"   NEURAL OS - Running: {cmd}")
    print("=" * 70)

    nos = NeuralOS()

    # Boot
    print("\nBooting...")
    nos.run(100000)

    # Run command
    print(f"Executing '{cmd}'...")
    nos.set_input(cmd)
    nos.run(100000)

    print_framebuffer(nos.get_framebuffer(), f"Output: {cmd}")


def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else []

    if not args:
        # Just boot and show
        print("\n" + "=" * 70)
        print("   NEURAL OS - BOOT")
        print("=" * 70)

        nos = NeuralOS()
        print("\nBooting...")
        nos.run(100000)

        print_framebuffer(nos.get_framebuffer(), "Neural OS")

        print("\nCommands: python run_neural_os.py [doom|ls|help|benchmark]")

    elif args[0] == "benchmark":
        run_benchmark()

    elif args[0] == "doom":
        run_doom()

    elif args[0] in ["ls", "help", "cat"]:
        run_command(args[0])

    else:
        print(f"Unknown command: {args[0]}")
        print("Try: doom, ls, help, cat, benchmark")


if __name__ == "__main__":
    main()
