#!/usr/bin/env python3
import torch
import struct
import time
from batched_neural_cpu_optimized import BatchedNeuralCPU

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class SimpleNeuralOS:
    def __init__(self):
        print("=" * 80)
        print("🖥️  NeuralOS v1.0")
        print("=" * 80)
        print(f"Device: {device}\n")
        
        print("Initializing Neural CPU...")
        self.cpu = BatchedNeuralCPU(memory_size=64*1024*1024, batch_size=128)
        print()
        
        self.running = False
        self.uptime_start = None
        self.programs = {}
        
        self._build_programs()
    
    def _build_programs(self):
        print("Building programs...")
        self.programs['counter'] = self._create_counter()
        print("   ✅ counter")
        self.programs['adder'] = self._create_adder()
        print("   ✅ adder")
        print()
    
    def _create_counter(self):
        # Simple counter using ADD instructions
        code = []
        # MOVZ X0, #0
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0 << 5) | 0))
        # ADD X0, X0, #1 (20 times)
        for _ in range(20):
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (1 << 10) | (0 << 5) | 0))
        return b''.join(code)
    
    def _create_adder(self):
        # Simple adder program
        code = []
        # MOVZ X0, #10
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (10 << 5) | 0))
        # MOVZ X1, #20
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (20 << 5) | 1))
        # MOVZ X2, #30
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (30 << 5) | 2))
        # ADD X3, X0, X1
        code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (1 << 10) | (0 << 5) | 3))
        # ADD X4, X2, X3
        code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (3 << 10) | (2 << 5) | 4))
        # ADD X5, X4, #5
        code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (5 << 10) | (4 << 5) | 5))
        return b''.join(code)
    
    def boot(self):
        print("=" * 80)
        print("🚀 Booting NeuralOS...")
        print("=" * 80)
        
        start = time.time()
        kernel = self._create_kernel()
        self.cpu.load_binary(kernel, 0x1000)
        results = self.cpu.run(max_instructions=len(kernel)//4)
        boot_time = time.time() - start
        
        print()
        print("✅ Boot Complete!")
        print(f"Time: {boot_time*1000:.1f}ms")
        print(f"Instructions: {results['instructions']}")
        print(f"IPS: {results['ips']:.0f}")
        print(f"Batches: {results['batches']}")
        print()
        
        self.uptime_start = time.time()
        self.running = True
        return True
    
    def _create_kernel(self):
        # Minimal kernel - just setup
        code = []
        # MOVZ X0, #0
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0 << 5) | 0))
        # MOVZ X1, #1
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (1 << 5) | 1))
        # MOVZ X2, #2
        code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (2 << 5) | 2))
        # Some initial operations
        for i in range(20):
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (i << 10) | (i % 3 << 5) | i % 3))
        return b''.join(code)
    
    def run_program(self, name):
        if name not in self.programs:
            print(f"❌ Unknown program: {name}")
            print(f"Available: {', '.join(sorted(self.programs.keys()))}")
            return False
        
        print(f"\n🔄 Running {name}...")
        prog = self.programs[name]
        self.cpu.load_binary(prog, 0x20000)
        self.cpu.pc.fill_(0x20000)
        results = self.cpu.run(max_instructions=300)
        
        print(f"✅ Done: {results['instructions']} instructions")
        print(f"   IPS: {results['ips']:.0f}")
        print(f"   Batches: {results['batches']}")
        
        print("\n📊 Registers:")
        for i in range(min(8, 32)):
            val = 0
            for b in range(64):
                if self.cpu.registers[i, b].item() > 0.5:
                    val |= (1 << b)
            if val > 0:
                print(f"   X{i}: {val}")
        return True
    
    def show_stats(self):
        if not self.uptime_start:
            return
        uptime = time.time() - self.uptime_start
        stats = self.cpu.stats
        print("\n📊 Stats:")
        print(f"Uptime: {uptime:.1f}s")
        print(f"Instructions: {stats['instructions']}")
        print(f"Batches: {stats['batches']}")
        print(f"IPS: {stats['instructions']/uptime:.0f}")
        print(f"Device: {device}")
        print()
    
    def shell(self):
        print("\n╔" + "═"*78 + "╗")
        print("║       Welcome to NeuralOS v1.0                                    ║")
        print("║           Running on MPS (Apple Silicon)                         ║")
        print("╚" + "═"*78 + "╝")
        print("Commands: help, stats, programs, run <name>, exit\n")
        
        while self.running:
            try:
                cmd = input("neural_os> ").strip().lower()
                if not cmd:
                    continue
                if cmd == 'help':
                    print("\nCommands: help, stats, programs, run <name>, exit\n")
                elif cmd == 'stats':
                    self.show_stats()
                elif cmd == 'programs':
                    print(f"\nAvailable: {', '.join(sorted(self.programs.keys()))}\n")
                elif cmd in ['exit', 'quit']:
                    print("Shutting down...")
                    self.running = False
                elif cmd in self.programs:
                    self.run_program(cmd)
                elif cmd.startswith('run '):
                    prog = cmd[4:].strip()
                    if prog in self.programs:
                        self.run_program(prog)
                else:
                    print(f"Unknown: {cmd}")
            except (KeyboardInterrupt, EOFError):
                print("\nShutting down...")
                self.running = False

def main():
    print("\n╔" + "═"*78 + "╗")
    print("║" + " "*20 + "🖥️  NeuralOS v1.0" + " "*38 + "║")
    print("║" + " "*10 + "Operating System for Neural Computing" + " "*22 + "║")
    print("╚" + "═"*78 + "╝\n")
    
    os = SimpleNeuralOS()
    if os.boot():
        os.shell()
    print("\nNeuralOS halted. Goodbye!")

if __name__ == "__main__":
    main()
