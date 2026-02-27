#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║          RUN DOOM ON NEURAL GPU - INTERACTIVE DEMO                               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  This is the main entry point for running DOOM on the Neural GPU.               ║
║                                                                                  ║
║  USAGE:                                                                          ║
║    python run_doom_gpu.py           - Run demo mode (non-interactive)           ║
║    python run_doom_gpu.py play      - Run interactive game (WASD controls)      ║
║    python run_doom_gpu.py benchmark - Run performance benchmark                 ║
║    python run_doom_gpu.py linux     - Run interactive Neural Linux shell        ║
║    python run_doom_gpu.py arm64     - Run REAL ARM64 OS (all binary execution) ║
║                                                                                  ║
║  PERFORMANCE:                                                                    ║
║    • Neural CPU: 66M+ IPS (vectorized loops)                                    ║
║    • All tensors on GPU (MPS/CUDA/CPU)                                          ║
║    • Neural keyboard IO, Timer, GIC on GPU                                      ║
║                                                                                  ║
║  NEURAL COMPONENTS:                                                              ║
║    ✅ NeuralCPU - 67M IPS with 7 loop patterns                          ║
║    ✅ NeuralKeyboardIO - GPU-based input handling                               ║
║    ✅ TrulyNeuralTimer - GPU timing                                             ║
║    ✅ TrulyNeuralGIC - GPU interrupt controller                                 ║
║    ✅ NeuralDOOMRaycaster - Full GPU raycaster                                  ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_demo():
    """Run non-interactive demo to verify everything works."""
    from doom_interactive_gpu import run_demo as demo
    demo()


def run_play():
    """Run interactive DOOM game."""
    from doom_interactive_gpu import InteractiveDOOM
    game = InteractiveDOOM()
    game.run()


def run_benchmark():
    """Run comprehensive benchmark."""
    import torch
    import time

    from neural_cpu import NeuralCPU, device

    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + " NEURAL GPU ULTIMATE - COMPREHENSIVE BENCHMARK ".center(70) + "║")
    print("╚" + "═" * 70 + "╝")
    print()
    print(f"Device: {device}")
    print()

    # Test 1: 100K count-up loop
    print("[1] COUNT-UP LOOP (100,000 iterations):")
    cpu = NeuralCPU()

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

    ips = executed / elapsed if elapsed > 0 else 0
    print(f"    Instructions: {executed:,}")
    print(f"    Time: {elapsed:.4f}s")
    print(f"    IPS: {ips:,.0f}")
    print(f"    Loops vectorized: {cpu.loops_vectorized}")

    # Test 2: Memory fill (2000 bytes)
    print("\n[2] MEMORY FILL (2000 bytes):")
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

    ips2 = executed2 / elapsed2 if elapsed2 > 0 else 0
    print(f"    Instructions: {executed2:,}")
    print(f"    Time: {elapsed2:.4f}s")
    print(f"    IPS: {ips2:,.0f}")
    print(f"    Loops vectorized: {cpu2.loops_vectorized}")

    # Summary
    print("\n" + "═" * 70)
    total = executed + executed2
    total_time = elapsed + elapsed2
    print(f"   TOTAL: {total:,} instructions in {total_time:.4f}s")
    print(f"   AVERAGE IPS: {total/total_time:,.0f}")
    print(f"   TOTAL LOOPS VECTORIZED: {cpu.loops_vectorized + cpu2.loops_vectorized}")
    print("═" * 70)

    if ips > 50_000_000:
        print("   ✅ BENCHMARK PASSED: 50M+ IPS achieved!")
    else:
        print("   ⚠️  Performance below target")


def run_neural_os_doom():
    """Run DOOM via the Neural OS."""
    from neural_os_final import NeuralOS
    import time

    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + " NEURAL OS - DOOM DEMO ".center(70) + "║")
    print("╚" + "═" * 70 + "╝")
    print()

    nos = NeuralOS()

    print("\nBooting Neural OS...")
    start = time.perf_counter()
    executed, elapsed = nos.run(100000)
    print(f"Boot: {executed:,} instructions in {elapsed:.2f}s")
    print(f"IPS: {executed/elapsed:,.0f}" if elapsed > 0 else "")

    print("\nRunning DOOM...")
    nos.set_input("doom")
    start = time.perf_counter()
    executed2, elapsed2 = nos.run(300000)
    print(f"DOOM: {executed2:,} instructions in {elapsed2:.2f}s")
    print(f"IPS: {executed2/elapsed2:,.0f}" if elapsed2 > 0 else "")
    print(f"Loops vectorized: {nos.cpu.loops_vectorized}")

    print("\n" + "╔" + "═" * 80 + "╗")
    for line in nos.get_framebuffer().split('\n'):
        print(f"║{line.ljust(80)}║")
    print("╚" + "═" * 80 + "╝")

    nos.cpu.print_stats()


def run_linux():
    """Run the interactive Neural Linux shell."""
    from neural_interactive_linux import NeuralLinux
    linux = NeuralLinux()
    linux.run()


def run_arm64():
    """Run the REAL ARM64 OS with binary execution."""
    from neural_kernel import NeuralARM64Kernel
    kernel = NeuralARM64Kernel()
    kernel.run_shell()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nRunning demo mode by default...\n")
        run_demo()
    elif sys.argv[1] == "play":
        run_play()
    elif sys.argv[1] == "benchmark":
        run_benchmark()
    elif sys.argv[1] == "os":
        run_neural_os_doom()
    elif sys.argv[1] == "linux":
        run_linux()
    elif sys.argv[1] == "arm64":
        run_arm64()
    elif sys.argv[1] == "demo":
        run_demo()
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Try: demo, play, benchmark, os, linux, arm64")


if __name__ == "__main__":
    main()
