#!/usr/bin/env python3.13
"""
Interactive Neural RTOS - DEBUG MODE
====================================
Shows what's happening internally for debugging.
"""

import sys
import time
import threading
sys.path.insert(0, '.')
from run_neural_rtos_v2 import FullyNeuralCPU, load_elf

INPUT_BUFFER = 0x50008
FB_ADDR = 0x40000

def main():
    print("="*60)
    print("NEURAL RTOS - INTERACTIVE DEBUG MODE")
    print("="*60)
    print()

    cpu = FullyNeuralCPU(fast_mode=True, batch_size=128, use_native_math=True)
    entry = load_elf(cpu, 'arm64_doom/neural_rtos.elf')
    cpu.pc = entry
    cpu.predecode_code_segment(0x10000, 0x2000)

    # Warm up
    print("[BOOT] Warming up (50K instructions)...")
    start = time.time()
    for i in range(50000):
        cpu.step()
    elapsed = time.time() - start
    ips = cpu.inst_count / elapsed
    print(f"[BOOT] Ready! {cpu.inst_count:,} inst in {elapsed:.1f}s ({ips:.0f} IPS)")
    print(f"[BOOT] Current PC: 0x{cpu.pc:x}")
    print()

    # Check if we're waiting for input
    waiting_for_input = 0x111c8 <= cpu.pc <= 0x11200
    print(f"[DEBUG] Current PC: 0x{cpu.pc:x}")
    print(f"[DEBUG] Waiting range: 0x111c8-0x11200")
    print(f"[DEBUG] Waiting for input: {waiting_for_input}")
    print(f"[DEBUG] Input buffer at: 0x{INPUT_BUFFER:x}")
    print()

    # Check framebuffer
    print("[FRAMEBUFFER] Checking initial state...")
    has_output = False
    for y in range(25):
        row = ""
        for x in range(80):
            ch = cpu.memory.read8(FB_ADDR + y * 80 + x)
            if 32 <= ch <= 126:
                row += chr(ch)
        if row.strip():
            has_output = True
            break
    print(f"[FRAMEBUFFER] Has content: {has_output}")
    print()

    # Main loop
    iteration = 0
    last_fb_check = 0

    try:
        while True:
            # Run instructions in chunks
            for _ in range(10000):
                cpu.step()

            iteration += 1

            # Check framebuffer every 50K instructions
            if iteration % 5 == 0:
                print(f"[DEBUG] Iteration {iteration*10000:7,} inst, PC: 0x{cpu.pc:x}")

                # Show any framebuffer content
                for y in range(25):
                    row = ""
                    for x in range(80):
                        ch = cpu.memory.read8(FB_ADDR + y * 80 + x)
                        if 32 <= ch <= 126:
                            row += chr(ch)
                        else:
                            row += ' '
                    if row.strip():
                        print(f"  Line {y:2d}: {row.rstrip()}")

            # Check for input prompt (waiting in kb_readline)
            if 0x111c8 <= cpu.pc <= 0x11200:
                print(f"[INPUT] Waiting for command at 0x{cpu.pc:x}")
                try:
                    cmd = input("> ")

                    if cmd.lower() in ['quit', 'exit', 'q']:
                        print("[STOP] Shutting down...")
                        break

                    # Write to input buffer
                    print(f"[INPUT] You typed: '{cmd}'")
                    for i, c in enumerate(cmd + "\n"):
                        if i < 64:
                            cpu.memory.write8(INPUT_BUFFER + i, ord(c))
                            print(f"[INPUT]   Wrote 0x{INPUT_BUFFER+i:x} = 0x{ord(c):02x} ('{c}')")

                except (EOFError, KeyboardInterrupt):
                    print("\n[STOP] Interrupted")
                    break

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")

    print(f"\n[STATS] Total instructions: {cpu.inst_count:,}")
    print(f"[STATS] Final PC: 0x{cpu.pc:x}")

if __name__ == "__main__":
    main()
