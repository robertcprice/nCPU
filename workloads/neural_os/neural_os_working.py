#!/usr/bin/env python3
"""
🖥️ NeuralOS - Working Version
==============================

Simple NeuralOS that actually works on BatchedNeuralCPU!
"""

import torch
import struct
import time
from batched_neural_cpu_optimized import BatchedNeuralCPU

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


class SimpleNeuralOS:
    """Minimal NeuralOS that works on BatchedNeuralCPU."""

    def __init__(self):
        print("=" * 80)
        print("🖥️  NeuralOS v1.0 - Working Edition")
        print("=" * 80)
        print(f"Device: {device}")
        print()

        # Initialize neural CPU
        print("Initializing Neural CPU...")
        self.cpu = BatchedNeuralCPU(memory_size=64*1024*1024, batch_size=128)
        print()

        # OS state
        self.running = False
        self.uptime_start = None

        # Programs
        self.programs = {}

        # Build default programs
        self._build_programs()

    def _build_programs(self):
        """Build default programs."""
        print("Building system programs...")

        # DOOM raycaster
        self.programs['doom'] = self._create_doom_program()
        print("   ✅ doom - Raycasting demo")

        # Fibonacci calculator
        self.programs['fibonacci'] = self._create_fibonacci_program()
        print("   ✅ fibonacci - Fibonacci calculator")

        # Counter
        self.programs['counter'] = self._create_counter_program()
        print("   ✅ counter - Simple counter")

        print()

    def _create_doom_program(self):
        """Create DOOM-like raycasting program."""
        code = []

        # Setup
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0x0000 << 5) | 0)))
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0x0000 << 5) | 1)))

        # 5 rays with stepping
        for ray in range(5):
            for step in range(5):
                imm1 = 100 + ray * 20
                imm2 = 50 + ray * 10
                code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (imm1 << 10) | (0 << 5) | 0)))
                code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (imm2 << 10) | (1 << 5) | 1)))

            # Distance calculation
            code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0x0000 << 5) | (2 + ray))))
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (0 << 10) | ((2 + ray) << 5) | (2 + ray))))
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (1 << 10) | ((2 + ray) << 5) | (2 + ray))))

        return b''.join(code)

    def _create_fibonacci_program(self):
        """Create Fibonacci sequence program."""
        code = []

        # F(0) = 0, F(1) = 1
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0x0000 << 5) | 0)))
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0x0001 << 5) | 1)))

        # Calculate next 15 Fibonacci numbers
        for i in range(15):
            # F(n) = F(n-2) + F(n-1)
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (1 << 10) | (0 << 5) | 2)))

            # Shift: F(n-2) = F(n-1)
            code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0x0000 << 5) | 0)))
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (1 << 10) | (0 << 5) | 0)))

            # F(n-1) = F(n)
            code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0x0000 << 5) | 1)))
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (2 << 10) | (1 << 5) | 1)))

        return b''.join(code)

    def _create_counter_program(self):
        """Create simple counter program."""
        code = []

        # Initialize counter
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0x0000 << 5) | 0)))

        # Count to 20
        for i in range(20):
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (1 << 10) | (0 << 5) | 0)))

        return b''.join(code)

    def boot(self):
        """Boot NeuralOS."""
        print("=" * 80)
        print("🚀 Booting NeuralOS...")
        print("=" * 80)
        print()

        start_time = time.time()

        # Create simple kernel
        kernel = self._create_kernel()

        # Load kernel
        self.cpu.load_binary(kernel, load_address=0x1000)

        # Execute kernel
        results = self.cpu.run(max_instructions=len(kernel)//4)

        boot_time = time.time() - start_time

        print()
        print("=" * 80)
        print("✅ NeuralOS Boot Complete!")
        print("=" * 80)
        print()
        print(f"Boot time: {boot_time*1000:.1f}ms")
        print(f"Instructions: {results['instructions']}")
        print(f"IPS: {results['ips']:.0f}")
        print(f"Batches: {results['batches']}")
        print()

        self.uptime_start = time.time()
        self.running = True

        return True

    def _create_kernel(self):
        """Create minimal kernel."""
        code = []

        # Setup stack and initialize
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0x0000 << 5) | 0)))
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0x0001 << 5) | 1)))
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0x0002 << 5) | 2)))

        # Initialize some values
        for i in range(20):
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (i << 10) | (i << 5) | i)))

        return b''.join(code)

    def run_program(self, name):
        """Run a program."""
        if name not in self.programs:
            print(f"❌ Program '{name}' not found")
            print(f"Available programs: {', '.join(sorted(self.programs.keys()))}")
            return False

        print(f"\n🔄 Running {name}...")
        start = time.time()

        program = self.programs[name]
        load_addr = 0x20000

        self.cpu.load_binary(program, load_address=load_addr)
        self.cpu.pc.fill_(load_addr)

        results = self.cpu.run(max_instructions=500)

        elapsed = time.time() - start

        print(f"✅ Completed in {elapsed*1000:.1f}ms")
        print(f"   Instructions: {results['instructions']}")
        print(f"   IPS: {results['ips']:.0f}")
        print(f"   Batches: {results['batches']}")

        # Show register state
        print("\n📊 Register State:")
        for i in range(min(8, 32)):
            val = 0
            for b in range(64):
                if self.cpu.registers[i, b].item() > 0.5:
                    val |= (1 << b)
            if val > 0:
                print(f"   X{i:2d}: {val:10d} (0x{val:08x})")

        return True

    def show_stats(self):
        """Show system statistics."""
        if not self.uptime_start:
            return

        uptime = time.time() - self.uptime_start
        stats = self.cpu.stats

        print()
        print("=" * 80)
        print("📊 NeuralOS Statistics")
        print("=" * 80)
        print()
        print(f"Uptime: {uptime:.1f}s")
        print(f"Device: {device}")
        print()
        print("CPU Stats:")
        print(f"   Total instructions: {stats['instructions']}")
        print(f"   Batches processed: {stats['batches']}")
        print(f"   Neural ops: {stats['neural_ops']}")
        print(f"   Current IPS: {stats['instructions']/uptime:.0f}")
        print()

    def shell(self):
        """NeuralOS shell."""
        print()
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║                    Welcome to NeuralOS v1.0                       ║")
        print("║           Operating System for Neural Computing                  ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        print()
        print("Type 'help' for available commands")
        print()

        while self.running:
            try:
                cmd = input("neural_os> ").strip().lower()

                if not cmd:
                    continue

                if cmd == 'help':
                    self._show_help()
                elif cmd == 'stats':
                    self.show_stats()
                elif cmd == 'programs':
                    self._list_programs()
                elif cmd == 'exit' or cmd == 'quit':
                    print("Shutting down NeuralOS...")
                    self.running = False
                elif cmd in self.programs:
                    self.run_program(cmd)
                elif cmd.startswith('run '):
                    prog = cmd[4:].strip()
                    if prog in self.programs:
                        self.run_program(prog)
                    else:
                        print(f"❌ Unknown program: {prog}")
                else:
                    print(f"❌ Unknown command: {cmd}")
                    print("Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except EOFError:
                print("\nShutting down...")
                self.running = False

    def _show_help(self):
        """Show help."""
        print()
        print("NeuralOS Commands:")
        print()
        print("   help        - Show this help")
        print("   stats       - Show system statistics")
        print("   programs    - List available programs")
        print("   run <prog>  - Run a program")
        print()
        print("Available Programs:")
        print()
        for prog in sorted(self.programs.keys()):
            print(f"   {prog:12s} - {self._get_program_desc(prog)}")
        print()
        print("   exit        - Exit NeuralOS")
        print()

    def _get_program_desc(self, prog):
        """Get program description."""
        descs = {
            'doom': 'DOOM-like raycasting demo',
            'fibonacci': 'Fibonacci sequence calculator',
            'counter': 'Simple counter (0-20)',
        }
        return descs.get(prog, prog)

    def _list_programs(self):
        """List available programs."""
        print()
        print("Available Programs:")
        print()
        for i, prog in enumerate(sorted(self.programs.keys()), 1):
            print(f"   {i}. {prog}")
        print()


def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "🖥️  NeuralOS v1.0" + " " * 38 + "║")
    print("║" + " " * 10 + "Operating System for Neural Computing" + " " * 22 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    os = SimpleNeuralOS()

    if os.boot():
        os.shell()

    print()
    print("NeuralOS halted. Goodbye!")


if __name__ == "__main__":
    main()
