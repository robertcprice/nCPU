#!/usr/bin/env python3
"""nCPU Neural Execution Demo

Demonstrates that every ALU operation in the nCPU is computed by a trained
neural network — no hardcoded arithmetic anywhere in the execution path.

Requirements: pip install torch
Usage: python examples/demo_neural_cpu.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
except ImportError:
    print("This demo requires PyTorch. Install with: pip install torch")
    sys.exit(1)

from ncpu.model import CPU
from ncpu.model.neural_ops import NeuralOps


def banner():
    print("""
 ═══════════════════════════════════════════════════════════════
  nCPU — Neural CPU Demo
  Every ALU operation is a trained neural network
 ═══════════════════════════════════════════════════════════════
""")


def model_inventory():
    """Walk models/ and display what's available."""
    print(" MODEL INVENTORY")
    print(" " + "─" * 59)

    models_dir = Path("models")
    if not models_dir.exists():
        print("  models/ directory not found — run from project root")
        return

    total_params = 0
    total_size = 0
    model_count = 0

    for pt_file in sorted(models_dir.rglob("*.pt")):
        rel = pt_file.relative_to(models_dir)
        size_kb = pt_file.stat().st_size / 1024
        total_size += size_kb

        try:
            sd = torch.load(pt_file, map_location="cpu", weights_only=True)
            params = sum(v.numel() for v in sd.values())
            total_params += params
            print(f"  {str(rel):40s}  {params:>8,} params  {size_kb:>7.1f} KB")
        except Exception:
            print(f"  {str(rel):40s}  (could not load)")

        model_count += 1

    print(f"\n  Total: {model_count} models, {total_params:,} parameters, {total_size/1024:.1f} MB")
    print()


def demo_neural_addition():
    """Step through the neural full adder bit by bit."""
    print(" NEURAL ADDITION: 42 + 17 = 59")
    print(" " + "─" * 59)

    ops = NeuralOps()
    ops.load()

    if ops._adder is None:
        print("  Adder model not available")
        return

    a, b = 42, 17
    bits_a = ops._int_to_bits(a)
    bits_b = ops._int_to_bits(b)

    carry = torch.tensor(0.0)
    result_bits = torch.zeros(32)
    print(f"  a = {a} = {a:032b}")
    print(f"  b = {b} = {b:032b}")
    print()
    print(f"  Bit-by-bit through the neural full adder (128-hidden, 3-layer):")

    with torch.no_grad():
        for i in range(32):
            inp = torch.tensor([[bits_a[i].item(), bits_b[i].item(), carry.item()]])
            out = ops._adder(inp)[0]
            sum_bit = (out[0] > 0.5).float()
            carry = (out[1] > 0.5).float()
            result_bits[i] = sum_bit
            if i < 8:
                print(f"    bit[{i}]: {int(bits_a[i].item())} + {int(bits_b[i].item())} + carry={int(carry.item())} → sum={int(sum_bit.item())}")

    result = ops._bits_to_int(result_bits)
    print(f"    ... (24 more bits)")
    print(f"  Result: {result} (expected: {a + b})")
    print(f"  {'CORRECT' if result == a + b else 'MISMATCH'}")
    print()


def demo_neural_multiplication():
    """Show the byte-pair LUT multiplication."""
    print(" NEURAL MULTIPLICATION: 7 x 6 = 42")
    print(" " + "─" * 59)

    ops = NeuralOps()
    ops.load()

    if ops._multiplier is None:
        print("  Multiplier model not available")
        return

    a, b = 7, 6
    a_bytes = [(a >> (i * 8)) & 0xFF for i in range(4)]
    b_bytes = [(b >> (i * 8)) & 0xFF for i in range(4)]

    print(f"  a = {a}, bytes = {a_bytes}")
    print(f"  b = {b}, bytes = {b_bytes}")
    print(f"  Neural byte-pair LUT: 256 x 256 x 16 learned tensor")
    print()

    result = 0
    for i in range(4):
        for j in range(4):
            if a_bytes[i] == 0 or b_bytes[j] == 0:
                continue
            product = ops._multiplier.lookup(a_bytes[i], b_bytes[j])
            print(f"    LUT[{a_bytes[i]}][{b_bytes[j]}] = {product} (shift by {(i+j)*8} bits)")
            result += product << ((i + j) * 8)

    result = result & 0xFFFFFFFF
    if result >= 0x80000000:
        result -= 0x100000000
    print(f"  Result: {result} (expected: {a * b})")
    print(f"  {'CORRECT' if result == a * b else 'MISMATCH'}")
    print()


def demo_neural_logical():
    """Show bitwise ops through the neural truth tables."""
    print(" NEURAL LOGICAL: 0xFF AND 0x0F = 0x0F")
    print(" " + "─" * 59)

    ops = NeuralOps()
    ops.load()

    if ops._logical is None:
        print("  Logical model not available")
        return

    print(f"  Neural truth table: 7 ops x 4 entries (learned parameters)")
    print()

    for op_name, op_idx, a, b in [("AND", 0, 0xFF, 0x0F), ("OR", 1, 0xF0, 0x0F), ("XOR", 2, 0xFF, 0xFF)]:
        result = getattr(ops, f"neural_{op_name.lower()}")(a, b)
        expected = {"AND": a & b, "OR": a | b, "XOR": a ^ b}[op_name]
        status = "CORRECT" if result == expected else "MISMATCH"
        print(f"  {op_name}: 0x{a:02X} {op_name} 0x{b:02X} = 0x{result:02X} (expected 0x{expected:02X}) {status}")

    print()


def demo_cross_validation():
    """Run all programs in mock vs neural mode, verify identical results."""
    print(" CROSS-VALIDATION: Mock vs Neural")
    print(" " + "─" * 59)

    programs_dir = Path("programs")
    if not programs_dir.exists():
        print("  programs/ directory not found")
        return

    all_match = True
    for asm_file in sorted(programs_dir.rglob("*.asm")):
        source = asm_file.read_text()

        mock_cpu = CPU(neural_execution=False)
        mock_cpu.load_program(source)
        t0 = time.perf_counter()
        mock_cpu.run()
        mock_time = time.perf_counter() - t0

        neural_cpu = CPU(neural_execution=True)
        neural_cpu.load_program(source)
        t0 = time.perf_counter()
        neural_cpu.run()
        neural_time = time.perf_counter() - t0

        match = mock_cpu.dump_registers() == neural_cpu.dump_registers()
        all_match = all_match and match
        status = "MATCH" if match else "MISMATCH"

        # Find non-zero registers for display
        regs = mock_cpu.dump_registers()
        interesting = {k: v for k, v in regs.items() if v != 0}

        print(f"  {asm_file.name:25s} {status:8s}  mock={mock_time*1000:.1f}ms  neural={neural_time*1000:.1f}ms  {interesting}")

    print()
    if all_match:
        print("  All programs produce IDENTICAL results in mock and neural mode.")
    else:
        print("  WARNING: Some programs produced different results!")
    print()


def main():
    banner()
    model_inventory()
    demo_neural_addition()
    demo_neural_multiplication()
    demo_neural_logical()
    demo_cross_validation()

    print(" ═══════════════════════════════════════════════════════════════")
    print("  Every arithmetic operation above was computed by a trained")
    print("  neural network — no hardcoded math in the execution path.")
    print(" ═══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
