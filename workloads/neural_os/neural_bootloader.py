#!/usr/bin/env python3
"""
🖥️ Neural Bootloader - Multi-OS/Program Selection System
==========================================================

A bootloader that lets you select which OS or program to run on the neural CPU.

Features:
- Multiple OS environments (NeuralOS, DOOM, benchmarks)
- Interactive program selection
- Quick testing mode
- Performance statistics
"""

import torch
import struct
import time
import sys
from batched_neural_cpu_optimized import BatchedNeuralCPU

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class NeuralBootloader:
    """Multi-OS bootloader for neural computing systems."""

    def __init__(self):
        print("=" * 80)
        print("🔧 Neural Bootloader v1.0")
        print("=" * 80)
        print(f"Device: {device}\n")

        print("Initializing Neural CPU...")
        self.cpu = BatchedNeuralCPU(memory_size=64*1024*1024, batch_size=128)
        print()

        self.programs = {}
        self._build_all_programs()

    def _build_all_programs(self):
        """Build all available programs across different environments."""
        print("Building program library...")

        # ===== NeuralOS Programs =====
        print("\n📦 NeuralOS:")
        self.programs['neuralos:counter'] = self._create_counter()
        print("   ✅ counter - Simple counter (0-20)")

        self.programs['neuralos:adder'] = self._create_adder()
        print("   ✅ adder - Arithmetic test")

        self.programs['neuralos:fibonacci'] = self._create_fibonacci()
        print("   ✅ fibonacci - Fibonacci sequence")

        # ===== DOOM Programs =====
        print("\n🎮 DOOM:")
        self.programs['doom:raycaster'] = self._create_doom_raycaster()
        print("   ✅ raycaster - DOOM-like raycasting (7 FPS)")

        # ===== Benchmark Programs =====
        print("\n⚡ Benchmarks:")
        self.programs['bench:alu'] = self._create_alu_benchmark()
        print("   ✅ alu - ALU performance test")

        self.programs['bench:memory'] = self._create_memory_benchmark()
        print("   ✅ memory - Memory access test")

        print(f"\n✅ Total programs: {len(self.programs)}\n")

    # ===== Program Creators =====

    def _create_counter(self):
        """Simple counter program."""
        code = []
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0 << 5) | 0))  # MOVZ X0, #0
        for _ in range(20):
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (1 << 10) | (0 << 5) | 0))  # ADD X0, X0, #1
        return b''.join(code)

    def _create_adder(self):
        """Arithmetic test program."""
        code = []
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (10 << 5) | 0))  # MOVZ X0, #10
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (20 << 5) | 1))  # MOVZ X1, #20
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (30 << 5) | 2))  # MOVZ X2, #30
        code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (1 << 10) | (0 << 5) | 3))  # ADD X3, X0, X1
        code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (3 << 10) | (2 << 5) | 4))  # ADD X4, X2, X3
        code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (5 << 10) | (4 << 5) | 5))  # ADD X5, X4, #5
        return b''.join(code)

    def _create_fibonacci(self):
        """Fibonacci sequence calculator."""
        code = []
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0 << 5) | 0))  # MOVZ X0, #0 (F0)
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (1 << 5) | 1))  # MOVZ X1, #1 (F1)

        for _ in range(15):
            # F2 = F0 + F1
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (1 << 10) | (0 << 5) | 2))  # ADD X2, X0, X1
            # Shift: F0 = F1
            code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0 << 5) | 0))  # MOVZ X0, #0
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (1 << 10) | (0 << 5) | 0))  # ADD X0, X0, #1
            # F1 = F2
            code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0 << 5) | 1))  # MOVZ X1, #0
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (2 << 10) | (1 << 5) | 1))  # ADD X1, X1, #2

        return b''.join(code)

    def _create_doom_raycaster(self):
        """DOOM-like raycasting program."""
        code = []
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0 << 5) | 0))  # ray_x
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0 << 5) | 1))  # ray_y

        # 5 rays with raycasting
        for ray in range(5):
            for step in range(10):
                imm1 = 100 + ray * 20
                imm2 = 50 + ray * 10
                code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (imm1 << 10) | (0 << 5) | 0))
                code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (imm2 << 10) | (1 << 5) | 1))

            # Distance calculation
            target = 2 + ray
            code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0 << 5) | target))
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (0 << 10) | (target << 5) | target))
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (1 << 10) | (target << 5) | target))

        return b''.join(code)

    def _create_alu_benchmark(self):
        """ALU performance benchmark."""
        code = []
        # Initialize
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (100 << 5) | 0))
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (50 << 5) | 1))

        # 100 arithmetic operations
        for i in range(100):
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (i << 10) | (0 << 5) | 2))

        return b''.join(code)

    def _create_memory_benchmark(self):
        """Memory access benchmark."""
        code = []
        # Sequential access pattern
        for i in range(50):
            code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (i << 5) | i % 8))

        return b''.join(code)

    # ===== Execution =====

    def run_program(self, name):
        """Run a specific program."""
        if name not in self.programs:
            print(f"❌ Unknown program: {name}")
            return False

        print(f"\n🔄 Running {name}...")
        start = time.time()

        program = self.programs[name]
        load_addr = 0x20000

        self.cpu.load_binary(program, load_address=load_addr)
        self.cpu.pc.fill_(load_addr)

        results = self.cpu.run(max_instructions=1000)
        elapsed = time.time() - start

        print(f"✅ Completed in {elapsed*1000:.1f}ms")
        print(f"   Instructions: {results['instructions']}")
        print(f"   IPS: {results['ips']:.0f}")
        print(f"   Batches: {results['batches']}")

        return True

    # ===== Menu System =====

    def show_main_menu(self):
        """Display main menu."""
        print("\n" + "=" * 80)
        print("🖥️  NEURAL BOOTLOADER - Main Menu")
        print("=" * 80)
        print("\nSelect an environment:\n")

        categories = {
            '1': ('NeuralOS', 'Operating System with programs'),
            '2': ('DOOM', 'Game demos and raycasting'),
            '3': ('Benchmarks', 'Performance tests'),
            '4': ('All Programs', 'List all available programs'),
            '0': ('Exit', 'Shutdown system')
        }

        for key, (name, desc) in categories.items():
            print(f"   {key}. {name:15s} - {desc}")

        print()

    def show_category_menu(self, category):
        """Show programs in a category."""
        prefix = category.lower()

        print(f"\n{'=' * 80}")
        print(f"📦 {category.upper()} Programs")
        print(f"{'=' * 80}\n")

        matching = [(k, k.split(':', 1)[1]) for k in self.programs.keys() if k.startswith(prefix)]
        matching.sort()

        if not matching:
            print("No programs found.")
            return

        for i, (full_name, short_name) in enumerate(matching, 1):
            print(f"   {i}. {short_name}")

        print(f"\n   0. Back to main menu")
        print()

    def run_category_menu(self, category):
        """Handle category program selection."""
        prefix = category.lower()
        matching = [k for k in self.programs.keys() if k.startswith(prefix)]
        matching.sort()

        while True:
            self.show_category_menu(category)

            try:
                choice = input(f"{category}> ").strip()

                if choice == '0':
                    return

                idx = int(choice) - 1
                if 0 <= idx < len(matching):
                    prog_name = matching[idx]
                    self.run_program(prog_name)
                    input("\nPress Enter to continue...")
                else:
                    print("❌ Invalid selection")

            except (ValueError, KeyboardInterrupt, EOFError):
                print("\nReturning to main menu...")
                return

    # ===== Main Loop =====

    def main_loop(self):
        """Main bootloader loop."""
        while True:
            self.show_main_menu()

            try:
                choice = input("bootloader> ").strip()

                if choice == '0':
                    print("\n🛑 Shutting down...")
                    break
                elif choice == '1':
                    self.run_category_menu('NeuralOS')
                elif choice == '2':
                    self.run_category_menu('DOOM')
                elif choice == '3':
                    self.run_category_menu('Benchmarks')
                elif choice == '4':
                    print("\n📋 All Available Programs:\n")
                    for prog in sorted(self.programs.keys()):
                        print(f"   - {prog}")
                    print()
                else:
                    print("❌ Invalid selection")

            except (KeyboardInterrupt, EOFError):
                print("\n\n🛑 Shutdown initiated...")
                break


def main():
    print("\n╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "🔧 Neural Bootloader v1.0" + " " * 33 + "║")
    print("║" + " " * 10 + "Multi-OS/Program Selection System" + " " * 26 + "║")
    print("╚" + "═" * 78 + "╝\n")

    bootloader = NeuralBootloader()
    bootloader.main_loop()

    print("\n✅ System halted. Goodbye!\n")


if __name__ == "__main__":
    main()
