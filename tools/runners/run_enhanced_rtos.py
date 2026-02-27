#!/usr/bin/env python3.13
"""
Neural RTOS Enhanced - Interactive Shell
========================================
Enhanced interactive mode with persistent filesystem and more features.

Features:
- Persistent file storage (saved to .rtos_fs)
- Import/export files from host system
- Enhanced shell with Linux-like commands
- Linux syscall compatibility for running ARM64 binaries
- Robust error handling

Usage:
    python3.13 run_enhanced_rtos.py

Commands:
    ls, cat, touch, rm, edit, cp, mv
    import, export, stat, df
    echo, calc, mem, date, uname, hostname, pwd, env
    help, clear, reboot
"""

import sys
import time
import select
import tty
import termios
import os

sys.path.insert(0, '.')
from run_neural_rtos_v2 import FullyNeuralCPU, load_elf, FramebufferIO
from rtos_filesystem import get_filesystem
from rtos_shell import RTOSShell


def main():
    print("="*60)
    print("NEURAL RTOS - ENHANCED MODE")
    print("="*60)
    print("Initializing neural CPU...")

    cpu = FullyNeuralCPU(fast_mode=True, batch_size=128, use_native_math=True)
    fb = FramebufferIO(cpu)
    entry = load_elf(cpu, 'arm64_doom/neural_rtos.elf')
    cpu.pc = 0x10000  # Start from actual code, not entry point
    cpu.predecode_code_segment(0x10000, 0x2000)

    # Initialize filesystem and shell
    fs = get_filesystem()
    shell = RTOSShell(cpu, fb)

    # Show filesystem stats
    stats = fs.get_stats()
    print(f"[OK] Filesystem loaded: {stats['total_files']} files")

    # Set up terminal for non-blocking input
    old_settings = None
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    except:
        pass

    print("\n" + "="*60)
    print("NEURAL RTOS ENHANCED SHELL")
    print("="*60)
    print("Type 'help' for available commands")
    print("Press Ctrl+C to exit")
    print("="*60 + "\n")

    try:
        start_time = time.time()
        cmd_buffer = ""
        ps1 = "neural@rtos:~$ "

        while not cpu.halted:
            # Run CPU in small batches (even if stuck, we can still use shell)
            for _ in range(50):
                cpu.step()

            # Check for keyboard input (non-blocking)
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)

                # Handle Ctrl+C
                if ch == '\x03':
                    print("\nShutting down...")
                    break

                # Handle Enter
                if ch == '\r' or ch == '\n':
                    if cmd_buffer:
                        # Execute command through enhanced shell
                        output = shell.execute(cmd_buffer)

                        # Print output
                        if output:
                            # Handle clear screen
                            if output == "\033[2J\033[H":
                                print("\033[2J\033[H", end="")
                            else:
                                print(output, end="")

                        cmd_buffer = ""
                    print(ps1, end="", flush=True)

                # Handle Backspace
                elif ch == '\x7f' or ch == '\x08':
                    if cmd_buffer:
                        cmd_buffer = cmd_buffer[:-1]
                        sys.stdout.write('\b \b')
                        sys.stdout.flush()

                # Regular character
                else:
                    cmd_buffer += ch
                    sys.stdout.write(ch)
                    sys.stdout.flush()

            # Display framebuffer periodically
            now = time.time()
            if now - start_time > 0.05:
                frame = fb.read_frame()
                if any(line.strip() for line in frame):
                    print("\033[H", end="")
                    for line in frame:
                        print(line)
                start_time = now

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        if old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        # Save filesystem
        fs.save()

    elapsed = time.time() - start_time
    ips = cpu.inst_count / elapsed if elapsed > 0 else 0
    print(f"\nStats: {cpu.inst_count:,} total instructions executed")
    print(f"Final PC: 0x{cpu.pc:x}")
    print(f"Average IPS: {ips:.0f}")
    print(f"Filesystem: {stats['total_files']} files saved to {stats['storage_path']}")


if __name__ == "__main__":
    main()
