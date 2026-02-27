#!/usr/bin/env python3
"""
NEURAL OS - INTERACTIVE TERMINAL
================================
A REAL interactive terminal running on the Neural GPU CPU!

Commands:
  help     - Show available commands
  ls       - List files
  cat      - Show file contents
  doom     - Run DOOM raycaster
  clear    - Clear screen
  bench    - Run speed benchmark
  exit     - Exit terminal
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_os_final import NeuralOS, FB_WIDTH, FB_HEIGHT
from neural_cpu import NeuralCPU, device
import torch


class NeuralTerminal:
    def __init__(self):
        print("\033[2J\033[H")  # Clear screen
        print("=" * 70)
        print("   NEURAL OS - INTERACTIVE TERMINAL")
        print("   Running on Neural GPU Ultimate CPU (59M+ IPS vectorized!)")
        print("=" * 70)
        print()

        self.nos = NeuralOS()

        print("Booting Neural OS...")
        start = time.perf_counter()
        executed, _ = self.nos.run(100000)
        boot_time = time.perf_counter() - start
        print(f"Boot complete: {executed:,} instructions in {boot_time:.2f}s")

    def get_display(self):
        return self.nos.get_framebuffer()

    def send_command(self, cmd):
        self.nos.set_input(cmd)
        start = time.perf_counter()
        executed, _ = self.nos.run(500000)
        elapsed = time.perf_counter() - start
        return self.get_display(), executed, elapsed

    def run_benchmark(self):
        print("\n" + "=" * 60)
        print("   NEURAL CPU SPEED BENCHMARK")
        print("=" * 60)

        cpu = NeuralCPU()

        code = bytearray()
        code.extend((0xD2800000).to_bytes(4, 'little'))
        code.extend((0xD290D401).to_bytes(4, 'little'))
        code.extend((0xF2A00021).to_bytes(4, 'little'))
        code.extend((0x91000400).to_bytes(4, 'little'))
        code.extend((0xEB01001F).to_bytes(4, 'little'))
        code.extend((0x54FFFFCB).to_bytes(4, 'little'))
        code.extend((0x00000000).to_bytes(4, 'little'))

        cpu.load_binary(bytes(code), 0)
        cpu.run(100)
        cpu.pc = torch.tensor(0, dtype=torch.int64, device=device)
        cpu.regs[0] = 0
        cpu.regs[1] = 0
        cpu.halted = False
        cpu.inst_count = torch.tensor(0, dtype=torch.int64, device=device)

        start = time.perf_counter()
        executed, _ = cpu.run(500000)
        elapsed = time.perf_counter() - start

        print(f"\n   100,000 iteration loop (VECTORIZED):")
        print(f"   Instructions: {executed:,}")
        print(f"   Time: {elapsed:.4f}s")
        print(f"   IPS: {executed/elapsed:,.0f}")
        print("=" * 60)

    def print_fb(self, fb, title=""):
        lines = fb.split('\n')
        print(f"\n┌─ {title} " + "─" * (72 - len(title)) + "┐")
        for line in lines:
            padded = line.ljust(FB_WIDTH)[:FB_WIDTH]
            print(f"│{padded}│")
        print("└" + "─" * 82 + "┘")

    def run(self):
        print("\nType 'help' for commands, 'exit' to quit\n")
        self.print_fb(self.get_display(), "Neural OS")

        while True:
            try:
                cmd = input("\n\033[1;32mneural>\033[0m ").strip()
                if not cmd:
                    continue

                if cmd in ["exit", "quit"]:
                    print("\nGoodbye!")
                    break

                if cmd in ["bench", "benchmark"]:
                    self.run_benchmark()
                    continue

                print(f"\nExecuting '{cmd}'...")
                fb, executed, elapsed = self.send_command(cmd)
                ips = executed / elapsed if elapsed > 0 else 0
                print(f"({executed:,} inst, {elapsed:.2f}s, {ips:,.0f} IPS, {self.nos.cpu.loops_vectorized} loops vectorized)")
                self.print_fb(fb, f"Output: {cmd}")

            except KeyboardInterrupt:
                print("\n\nUse 'exit' to quit")
            except EOFError:
                break


if __name__ == "__main__":
    NeuralTerminal().run()
