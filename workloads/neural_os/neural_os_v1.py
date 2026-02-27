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
        self.programs['fibonacci'] = self._create_fibonacci()
        print("   ✅ fibonacci")
        self.programs['doom'] = self._create_doom()
        print("   ✅ doom")
        print()
    
    def _create_counter(self):
        code = []
        # Init
        code.append(self._pack(0b10100, 0, 0, 0))  # MOVZ X0, #0
        # Count 0-19
        for i in range(20):
            code.append(self._pack(0b10000, 0, 0, 1))  # ADD X0, X0, #1
        return b''.join(code)
    
    def _create_fibonacci(self):
        code = []
        # F0=0, F1=1
        code.append(self._pack(0b10100, 0, 0, 0))
        code.append(self._pack(0b10100, 1, 1, 0))
        # 15 iterations
        for _ in range(15):
            code.append(self._pack(0b10000, 2, 0, 1))  # ADD X2, X0, X1
            code.append(self._pack(0b10100, 0, 0, 0))
            code.append(self._pack(0b10000, 0, 0, 1))
            code.append(self._pack(0b10100, 1, 0, 0))
            code.append(self._pack(0b10000, 1, 1, 2))
        return b''.join(code)
    
    def _create_doom(self):
        code = []
        # Init ray positions
        code.append(self._pack(0b10100, 0, 0, 0))  # ray_x
        code.append(self._pack(0b10100, 1, 0, 0))  # ray_y
        # 5 rays, 5 steps each
        for ray in range(5):
            for step in range(5):
                imm1 = 100 + ray * 20
                imm2 = 50 + ray * 10
                code.append(self._pack(0b10000, 0, 0, imm1))
                code.append(self._pack(0b10000, 1, 1, imm2))
            # Calc distance
            target = 2 + ray
            code.append(self._pack(0b10100, target, 0, 0))
            code.append(self._pack(0b10000, target, 0, 0))
            code.append(self._pack(0b10000, target, 1, target))
        return b''.join(code)
    
    def _pack(self, op, rd, rn, imm):
        return struct.pack('<I', (1 << 31) | (op << 24) | (imm << 10) | (rn << 5) | rd)
    
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
        print()
        
        self.uptime_start = time.time()
        self.running = True
        return True
    
    def _create_kernel(self):
        code = []
        # Setup
        code.append(self._pack(0b10100, 0, 0, 0))  # MOVZ X0, #0
        code.append(self._pack(0b10100, 1, 1, 0))  # MOVZ X1, #1
        code.append(self._pack(0b10100, 2, 2, 0))  # MOVZ X2, #2
        # Init
        for i in range(20):
            code.append(self._pack(0b10000, i, i, 0))  # ADD Xi, Xi, #0
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
        
        print("\n📊 Registers:")
        for i in range(8):
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
        print(f"IPS: {stats['instructions']/uptime:.0f}\n")
    
    def shell(self):
        print("\n╔" + "═"*78 + "╗")
        print("║       Welcome to NeuralOS v1.0                                    ║")
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
    print("╚" + "═"*78 + "╝\n")
    
    os = SimpleNeuralOS()
    if os.boot():
        os.shell()
    print("NeuralOS halted.")

if __name__ == "__main__":
    main()
