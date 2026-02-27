#!/usr/bin/env python3.13
"""
Interactive Neural RTOS
========================
Run this script to interact with the neural RTOS through your terminal.

Commands available:
- help    - Show help
- calc    - Calculator
- mem     - Show memory
- info    - Show system info
- ls      - List files
- cat     - Show file
"""

import sys
import time
import select
import tty
import termios
sys.path.insert(0, '.')
from run_neural_rtos_v2 import FullyNeuralCPU, load_elf, FramebufferIO, KeyboardIO

INPUT_BUFFER = 0x50008

def main():
    print("="*60)
    print("NEURAL RTOS - INTERACTIVE MODE")
    print("="*60)
    print("Initializing neural CPU...")

    cpu = FullyNeuralCPU(fast_mode=True, batch_size=128, use_native_math=True)
    fb = FramebufferIO(cpu)
    kb = KeyboardIO(cpu)
    entry = load_elf(cpu, 'arm64_doom/neural_rtos.elf')
    # Note: The ELF entry point (0x11280) points to empty memory
    # The actual code starts at 0x10000, so we start there
    cpu.pc = 0x10000
    cpu.predecode_code_segment(0x10000, 0x2000)

    # Set up terminal for non-blocking input
    old_settings = None
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    except:
        pass

    print("Booting... (Press Ctrl+C to exit)")
    time.sleep(1)

    try:
        start_time = time.time()
        last_display = 0
        cmd_buffer = ""

        while not cpu.halted:
            # Run in small batches
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
                    # Write command to RTOS input buffer
                    for i, c in enumerate(cmd_buffer + "\n"):
                        if i < 64:
                            cpu.memory.write8(INPUT_BUFFER + i, ord(c))
                    cmd_buffer = ""
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

            # Display framebuffer every 50ms
            now = time.time()
            if now - last_display > 0.05:
                fb.display()
                last_display = now

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        if old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    elapsed = time.time() - start_time
    ips = cpu.inst_count / elapsed
    print(f"\nStats: {cpu.inst_count:,} total instructions executed")
    print(f"Final PC: 0x{cpu.pc:x}")
    print(f"Average IPS: {ips:.0f}")

if __name__ == "__main__":
    main()
