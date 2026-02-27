#!/usr/bin/env python3
"""
INTERACTIVE DOOM WITH NEURAL KEYBOARD
======================================
Run DOOM rendering loops with neural optimizations and interactive keyboard control.

Controls:
- SPACE: Render one frame
- F: Run 10 frames
- R: Run until exit (benchmark mode)
- Q: Quit
- S: Toggle optimization on/off
- D: Show detailed stats
"""

import sys
import time
import termios
import tty
from pathlib import Path

sys.path.insert(0, '.')
from run_neural_rtos_v2 import FullyNeuralCPU, load_elf, KeyboardIO


def get_char():
    """Get a single character from stdin (non-blocking)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def set_nonblocking(enable=True):
    """Enable or disable non-blocking stdin."""
    import fcntl
    import os
    fd = sys.stdin.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    if enable:
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    else:
        fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)


def print_header(cpu):
    """Print the header."""
    print("\033[2J\033[H", end="")
    print("""
\033[36m╔══════════════════════════════════════════════════════════════════════════╗
║                    INTERACTIVE DOOM - NEURAL EDITION                      ║
║                    ====================================                    ║
║                                                                            ║
║  Controls: SPACE=1 frame  F=10 frames  R=run  S=toggle opt  Q=quit        ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════╝\033[0m
    """)

    opt_status = "\033[32mENABLED\033[0m" if cpu.enable_loop_optimization else "\033[31mDISABLED\033[0m"
    print(f"  Optimization: {opt_status}")
    print(f"  Entry point: 0x{cpu.pc:x}")
    print()


def print_stats(cpu, frame_time, frame_ips):
    """Print rendering statistics."""
    if cpu.loop_optimizer:
        stats = cpu.loop_optimizer.get_stats()

        print(f"\033[33m[FRAME STATS]\033[0m")
        print(f"  Time: {frame_time:.4f}s")
        print(f"  IPS: {frame_ips:,.0f}")
        print(f"  PC: 0x{cpu.pc:x}")
        print()

        print(f"\033[35m[OPTIMIZATION]\033[0m")
        print(f"  Loops detected: {stats['loops_detected']}")
        print(f"  Loops optimized: {stats['loops_optimized']}")
        print(f"  Loops rejected: {stats['loops_rejected']}")
        print(f"  Iterations saved: {stats['iterations_saved']:,}")

        if hasattr(cpu.loop_optimizer, 'pattern_discoverer'):
            discoverer = cpu.loop_optimizer.pattern_discoverer
            print(f"  Patterns discovered: {discoverer.stats['patterns_discovered']}")
            print(f"  Patterns matched: {discoverer.stats['patterns_matched']}")


def run_interactive_doom():
    """Run interactive DOOM with neural keyboard."""

    # Create CPU with optimization enabled by default
    print("\033[2J\033[H", end="")
    print("\033[36m[INIT]\033[0m Creating fully neural CPU...")
    cpu = FullyNeuralCPU(fast_mode=True, batch_size=128, use_native_math=True)

    # Load DOOM
    print("\033[36m[LOAD]\033[0m Loading DOOM benchmark...")
    entry = load_elf(cpu, 'doom_benchmark.elf')
    cpu.pc = entry
    print(f"   Entry point: 0x{entry:x}")

    # Enable optimization
    cpu.enable_optimization(enable=True)
    print("\033[32m[OK]\033[0m Neural loop optimization enabled")

    # Setup keyboard
    keyboard = KeyboardIO(cpu)

    # Instructions per frame (rough estimate based on loop sizes)
    INSTRUCTIONS_PER_FRAME = 5000

    # Print header
    print_header(cpu)

    # Main interactive loop
    total_instructions = 0
    start_time = time.time()
    running = True

    print("\033[33m[READY]\033[0m Press SPACE to render first frame, Q to quit")
    print()

    set_nonblocking(True)

    try:
        while running:
            # Check for keypress
            try:
                ch = sys.stdin.read(1)
            except:
                ch = None
                time.sleep(0.01)
                continue

            if ch:
                if ch == ' ' or ch == 'q' or ch == 'Q' or ch == 'r' or ch == 'R' or ch == 'f' or ch == 'F' or ch == 's' or ch == 'S' or ch == 'd' or ch == 'D':
                    # Handle these below
                    pass
                elif ch == '\x03':  # Ctrl+C
                    print("\n\033[33m[EXIT]\033[0m Interrupted")
                    break
                else:
                    continue

            # Process command
            frame_count = 0

            if ch == ' ':
                frame_count = 1
            elif ch == 'f' or ch == 'F':
                frame_count = 10
            elif ch == 'r' or ch == 'R':
                frame_count = -1  # Run until exit
            elif ch == 's' or ch == 'S':
                cpu.enable_optimization(not cpu.enable_loop_optimization)
                opt_status = "\033[32mENABLED\033[0m" if cpu.enable_loop_optimization else "\033[31mDISABLED\033[0m"
                print(f"\033[33m[TOGGLE]\033[0m Optimization now {opt_status}")
                print()
                continue
            elif ch == 'd' or ch == 'D':
                print_stats(cpu, 0, 0)
                print()
                print("\033[33m[PATTERNS]\033[0m Discovered patterns:")
                if hasattr(cpu.loop_optimizer, 'pattern_discoverer'):
                    for i, name in enumerate(cpu.loop_optimizer.pattern_discoverer.pattern_names):
                        print(f"  {name}")
                print()
                continue
            elif ch == 'q' or ch == 'Q':
                print("\n\033[33m[EXIT]\033[0m Quitting...")
                break
            else:
                continue

            # Render frames
            if frame_count != 0:
                if frame_count == -1:
                    print(f"\033[36m[RUNNING]\033[0m Running until exit (PC=0)...")
                    max_frames = 1000000
                else:
                    print(f"\033[36m[RENDERING]\033[0m {frame_count} frame(s)...")

                frame_start = time.time()

                for frame in range(frame_count if frame_count > 0 else max_frames):
                    # Run one frame worth of instructions
                    for _ in range(INSTRUCTIONS_PER_FRAME):
                        cpu.step()
                        if cpu.pc == 0:
                            print("\n\033[32m[EXIT]\033[0m Program terminated (PC=0)")
                            running = False
                            break
                        total_instructions += 1

                    if not running:
                        break

                    # Update progress
                    if frame_count > 0:
                        elapsed = time.time() - frame_start
                        ips = cpu.inst_count / elapsed if elapsed > 0 else 0
                        print(f"  Frame {frame + 1}/{frame_count} | PC=0x{cpu.pc:x} | IPS={ips:,.0f}\r", end="", flush=True)

                    if cpu.pc == 0:
                        break

                # Show stats
                frame_time = time.time() - frame_start
                frame_ips = total_instructions / frame_time if frame_time > 0 else 0

                print()
                print_stats(cpu, frame_time, frame_ips)
                print()

    except KeyboardInterrupt:
        print("\n\033[33m[EXIT]\033[0m Interrupted")
    finally:
        set_nonblocking(False)

    # Final stats
    total_time = time.time() - start_time
    avg_ips = total_instructions / total_time if total_time > 0 else 0

    print()
    print("=" * 78)
    print(" FINAL STATS")
    print("=" * 78)
    print(f"  Total instructions: {total_instructions:,}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average IPS: {avg_ips:,.0f}")
    print()

    if cpu.loop_optimizer:
        stats = cpu.loop_optimizer.get_stats()
        print("  Optimization Stats:")
        print(f"    Loops detected: {stats['loops_detected']}")
        print(f"    Loops optimized: {stats['loops_optimized']}")
        print(f"    Loops rejected: {stats['loops_rejected']}")
        print(f"    Iterations saved: {stats['iterations_saved']:,}")

    print("=" * 78)
    print()


if __name__ == "__main__":
    run_interactive_doom()
