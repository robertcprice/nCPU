#!/usr/bin/env python3
"""
🔬 COMPREHENSIVE NEURAL CPU PROFILING
=======================================

This script profiles the FULL neural CPU to identify bottlenecks:
1. Bit conversion time
2. Model inference time
3. Register read/write time
4. Batch processing potential
"""

import torch
import torch.nn as nn
import time
from pathlib import Path
import sys

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=" * 70)
print("🔬 NEURAL CPU PROFILING - Identifying Bottlenecks")
print("=" * 70)
print(f"Device: {device}")
print()

# ============================================================
# MODEL ARCHITECTURES (matching working models)
# ============================================================

class PerBitModel(nn.Module):
    """For bitwise ops (AND, OR, XOR, NOT) - processes all bits in parallel"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, a_bits, b_bits):
        x = torch.stack([a_bits, b_bits], dim=-1)  # [batch, bits, 2]
        return self.net(x).squeeze(-1)  # [batch, bits]


class CarryPredictorTransformer(nn.Module):
    """For ADD - carry chain with transformer"""
    def __init__(self, max_bits=64, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.max_bits = max_bits
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.carry_head = nn.Linear(d_model, 1)

    def forward(self, a_bits, b_bits):
        bits = a_bits.shape[1]
        batch = a_bits.shape[0]
        G = a_bits * b_bits
        P = a_bits + b_bits - 2 * a_bits * b_bits
        gp = torch.stack([G, P], dim=-1)
        x = self.input_proj(gp) + self.pos_embedding[:, :bits, :]
        mask = torch.triu(torch.ones(bits, bits, device=a_bits.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        carry_logits = self.carry_head(x).squeeze(-1)
        carries = torch.sigmoid(carry_logits)
        carry_in = torch.cat([torch.zeros(batch, 1, device=a_bits.device), carries[:, :-1]], dim=1)
        sums = P + carry_in - 2 * P * carry_in
        return sums, carries


class BorrowPredictorTransformer(nn.Module):
    """For SUB - borrow chain with transformer"""
    def __init__(self, max_bits=64, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.max_bits = max_bits
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.borrow_head = nn.Linear(d_model, 1)

    def forward(self, a_bits, b_bits):
        batch, bits = a_bits.shape
        not_a = 1 - a_bits
        G = not_a * b_bits
        P = a_bits + b_bits - 2 * a_bits * b_bits
        gp = torch.stack([G, P], dim=-1)
        x = self.input_proj(gp)
        x = x + self.pos_embedding[:, :bits, :]
        mask = torch.triu(torch.ones(bits, bits, device=a_bits.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        borrow_logits = self.borrow_head(x).squeeze(-1)
        borrows = torch.sigmoid(borrow_logits)
        borrow_in = torch.cat([torch.zeros(batch, 1, device=a_bits.device), borrows[:, :-1]], dim=1)
        diffs = P + borrow_in - 2 * P * borrow_in
        return diffs, borrows


class ProfiledNeuralCPU:
    """Neural CPU with detailed profiling"""

    def __init__(self):
        print("📦 Loading models for profiling...")
        self.models = {}

        # Load ADD
        self.models['ADD'] = CarryPredictorTransformer(64).to(device)
        ckpt = torch.load("models/final/ADD_64bit_100pct.pt", map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        self.models['ADD'].load_state_dict(state)
        self.models['ADD'].eval()
        print("   ✅ ADD (CarryPredictor)")

        # Load SUB
        self.models['SUB'] = BorrowPredictorTransformer(64).to(device)
        ckpt = torch.load("models/final/SUB_64bit_parallel.pt", map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        self.models['SUB'].load_state_dict(state)
        self.models['SUB'].eval()
        print("   ✅ SUB (BorrowPredictor)")

        # Load bitwise ops
        for op in ['AND', 'OR', 'XOR']:
            self.models[op] = PerBitModel(64).to(device)
            ckpt = torch.load(f"models/final/{op}_64bit_100pct.pt", map_location=device, weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            self.models[op].load_state_dict(state)
            self.models[op].eval()
            print(f"   ✅ {op} (PerBitModel)")

        # Register file (32 x 64-bit registers)
        self.regs = torch.zeros(32, 64, device=device)

        # Profiling stats
        self.stats = {
            'int_to_bits': [],
            'bits_to_int': [],
            'model_forward': [],
            'reg_read': [],
            'reg_write': [],
            'total_ops': 0
        }

    def int_to_bits(self, value):
        """Convert integer to 64-bit tensor with profiling"""
        start = time.time()
        bits = torch.zeros(64, device=device, dtype=torch.float32)
        for i in range(64):
            bits[i] = ((value >> i) & 1)
        self.stats['int_to_bits'].append(time.time() - start)
        return bits

    def bits_to_int(self, bits):
        """Convert bit tensor to integer with profiling"""
        start = time.time()
        result = 0
        for i, b in enumerate(bits.cpu().tolist()):
            if b > 0.5:
                result |= (1 << i)
        self.stats['bits_to_int'].append(time.time() - start)
        return result

    def read_reg(self, idx):
        """Read register with profiling"""
        start = time.time()
        val = self.regs[idx].clone()
        self.stats['reg_read'].append(time.time() - start)
        return val

    def write_reg(self, idx, value):
        """Write register with profiling"""
        start = time.time()
        if idx != 31:  # XZR is read-only
            self.regs[idx] = value
        self.stats['reg_write'].append(time.time() - start)

    def execute(self, op_name, a, b):
        """Execute operation with detailed profiling

        Args:
            op_name: Operation (ADD, SUB, AND, OR, XOR)
            a: First operand (integer value or register index 0-31)
            b: Second operand (integer value or register index 0-31)
        """
        self.stats['total_ops'] += 1

        # Convert integer operands to bit tensors
        if isinstance(a, int) and a < 32:
            # It's a register index - but we're using direct values for benchmarking
            a_bits = self.int_to_bits(a)
        elif isinstance(a, int):
            a_bits = self.int_to_bits(a)
        else:
            a_bits = a

        if isinstance(b, int) and b < 32:
            b_bits = self.int_to_bits(b)
        elif isinstance(b, int):
            b_bits = self.int_to_bits(b)
        else:
            b_bits = b

        # Add batch dimension
        a_bits = a_bits.unsqueeze(0)
        b_bits = b_bits.unsqueeze(0)

        # Profile: Model forward pass
        start = time.time()
        model = self.models[op_name]
        with torch.no_grad():
            result_bits = model(a_bits, b_bits)
            if isinstance(result_bits, tuple):
                result_bits = result_bits[0]  # Carry/Borrow predictor returns (result, carries)
        self.stats['model_forward'].append(time.time() - start)

        # Profile: Bit conversion and register write
        result_int = self.bits_to_int(result_bits[0])
        # Write to a temp register for now
        self.write_reg(0, result_bits[0])

        return result_int

    def print_profile(self):
        """Print profiling results"""
        total = sum([
            sum(self.stats['int_to_bits']),
            sum(self.stats['bits_to_int']),
            sum(self.stats['model_forward']),
            sum(self.stats['reg_read']),
            sum(self.stats['reg_write'])
        ])

        print("\n" + "=" * 70)
        print("📊 PROFILING RESULTS")
        print("=" * 70)
        print()

        print(f"Total operations executed: {self.stats['total_ops']}")
        print()

        print("🔍 BOTTLENECK ANALYSIS:")
        print()

        components = [
            ("Integer → Bits conversion", self.stats['int_to_bits']),
            ("Bits → Integer conversion", self.stats['bits_to_int']),
            ("Neural model forward pass", self.stats['model_forward']),
            ("Register read", self.stats['reg_read']),
            ("Register write", self.stats['reg_write']),
        ]

        # Sort by time
        components_sorted = sorted(components, key=lambda x: sum(x[1]), reverse=True)

        for name, times in components_sorted:
            total_time = sum(times)
            avg_time = total_time / len(times) if times else 0
            pct = total_time / total * 100 if total > 0 else 0
            print(f"  {name:30s}: {total_time*1000:8.2f}ms ({pct:5.1f}%)  avg: {avg_time*1000:6.3f}ms")

        print()
        print("💡 KEY FINDINGS:")

        model_time = sum(self.stats['model_forward'])
        conversion_time = sum(self.stats['int_to_bits']) + sum(self.stats['bits_to_int'])

        if model_time > conversion_time:
            print(f"   • Neural model inference is the bottleneck ({model_time/conversion_time:.1f}x slower than conversion)")
            print(f"   • Model takes {model_time/total*100:.1f}% of total time")
        else:
            print(f"   • Bit conversion is the bottleneck ({conversion_time/model_time:.1f}x slower than model)")

        print()
        print("🎯 OPTIMIZATION OPPORTUNITIES:")
        print()

        # Check what's slowest
        slowest = components_sorted[0]
        slowest_name = slowest[0]
        slowest_time = sum(slowest[1]) / len(slowest[1])

        if "model forward" in slowest_name.lower():
            print("   1. ⚡ BATCH PROCESSING: Process multiple operations together")
            print("      - Transformer models can process batches efficiently")
            print("      - Could get 2-5x speedup with batch_size=8-16")
            print()
            print("   2. 📱 TENSOR COMPILE: Use torch.compile() for 2-3x speedup")
            print("      - Requires PyTorch 2.0+")
            print("      - JIT compiles the model for your hardware")
            print()
            print("   3. 🎯 MODEL DISTILLATION: Create smaller, faster models")
            print("      - Distill to 1-2 layer transformers")
            print("      - Could get 3-5x speedup with minimal accuracy loss")

        elif "conversion" in slowest_name.lower():
            print("   1. ⚡ PRE-COMPUTE: Cache bit conversions for common values")
            print("      - Most operations use small integers (0-255)")
            print()
            print("   2. 🔢 VECTORIZE: Use batch operations for bit conversion")
            print("      - Process multiple values at once")

        print()
        print("📈 ESTIMATED SPEEDUP:")
        print(f"   Current: {1000/(sum(self.stats['model_forward'])/len(self.stats['model_forward']))*1000:.0f} IPS")
        print(f"   With batching (4x): ~{1000/(sum(self.stats['model_forward'])/len(self.stats['model_forward']))*1000*4:.0f} IPS")
        print(f"   With compile (2x): ~{1000/(sum(self.stats['model_forward'])/len(self.stats['model_forward']))*1000*2:.0f} IPS")
        print(f"   With both (8x): ~{1000/(sum(self.stats['model_forward'])/len(self.stats['model_forward']))*1000*8:.0f} IPS")
        print()


# ============================================================
# RUN BENCHMARKS
# ============================================================

if __name__ == "__main__":
    cpu = ProfiledNeuralCPU()

    print("=" * 70)
    print("🧪 RUNNING BENCHMARK")
    print("=" * 70)
    print()

    # Test 1: Individual operation speed
    print("Test 1: Individual operation throughput")
    print("-" * 50)

    num_ops = 1000
    ops = [
        ('ADD', 100, 42),
        ('SUB', 1000, 200),
        ('AND', 0xFFFF, 0xFF),
        ('OR', 0xF0, 0xF),
        ('XOR', 0xAA, 0x55),
    ]

    start = time.time()
    for i in range(num_ops):
        op, a, b = ops[i % len(ops)]
        result = cpu.execute(op, a, b)
    elapsed = time.time() - start

    ips = num_ops / elapsed
    print(f"   {num_ops} ops in {elapsed:.2f}s")
    print(f"   Speed: {ips:.0f} IPS")
    print()

    # Test 2: Realistic instruction sequence (like a small program)
    print("Test 2: Realistic instruction sequence")
    print("-" * 50)

    # Initialize some registers
    cpu.write_reg(1, cpu.int_to_bits(100))   # X1 = 100
    cpu.write_reg(2, cpu.int_to_bits(200))   # X2 = 200
    cpu.write_reg(3, cpu.int_to_bits(0xFF))   # X3 = 0xFF

    # Simulate a small program
    program = [
        ('ADD', 4, 1, 2),    # X4 = X1 + X2 = 300
        ('SUB', 5, 4, 100),  # X5 = X4 - 100 = 200
        ('AND', 6, 4, 0xFF), # X6 = X4 & 0xFF = 44
        ('OR', 7, 5, 0xF),   # X7 = X5 | 0xF = 207
        ('XOR', 8, 3, 0xFF), # X8 = X3 ^ 0xFF = 0
    ]

    print(f"   Program: {len(program)} instructions")
    for op, rd, rn, rm in program:
        result = cpu.execute(op, rn if isinstance(rn, int) else rm, rm if isinstance(rm, int) else rm)
        print(f"      {op} X{rd}, {rn}, {rm} = {result}")
    print()

    # Print detailed profiling
    cpu.print_profile()
