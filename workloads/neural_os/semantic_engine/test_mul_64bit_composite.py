#!/usr/bin/env python3
"""
Test 64-bit multiplication using 32-bit model composition.

Mathematical identity:
(A·2³² + B) × (C·2³² + D) = AC·2⁶⁴ + (AD + BC)·2³² + BD

Where A, B, C, D are 32-bit numbers.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_mul_add_hybrid import PureParallelMulWithADDv2


def load_32bit_model(checkpoint_path, device='cpu'):
    """Load the trained 32-bit multiplication model."""
    model = PureParallelMulWithADDv2(max_bits=64, d_model=384, nhead=16, num_layers=7)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model.to(device)


def int_to_bits(x, bits=32):
    """Convert integer to bit tensor."""
    return ((x.unsqueeze(-1) >> torch.arange(bits, device=x.device)) & 1).float()


def bits_to_int(bits):
    """
    Convert bit tensor to integer.

    IMPORTANT: Must use int64 for powers to avoid float32 precision loss!
    Float32 only has 24 bits of mantissa, so 2^25 and beyond are inexact.

    For >64 bits, we limit to first 64 (int64 max is 2^63-1, but unsigned range is 0 to 2^64-1)
    """
    num_bits = min(bits.shape[-1], 64)  # Limit to 64 bits to avoid overflow
    # Build powers carefully - int64 can hold up to 2^62 safely for multiplication
    # For 64-bit we'll sum directly to avoid overflow in intermediate
    result = torch.zeros(bits.shape[0], dtype=torch.int64, device=bits.device)
    for i in range(num_bits):
        result = result + (bits[:, i].long() << i)
    return result


def mul_32bit(model, a, b, device='cpu'):
    """
    Use trained model for 32-bit multiplication.

    IMPORTANT: The 32-bit model was trained on 16-bit × 16-bit → 32-bit.
    For compositional 64-bit, we need 32-bit × 32-bit → 64-bit.
    This requires the 64-bit model, not the 32-bit checkpoint!
    """
    # Prepare inputs - pad to 64 bits for model
    a_bits = int_to_bits(a, 64)
    b_bits = int_to_bits(b, 64)

    # Run model - get full 64-bit output
    with torch.no_grad():
        output = model(a_bits, b_bits)
        pred_bits = (torch.sigmoid(output[:, :64]) > 0.5).float()

    return bits_to_int(pred_bits)


def mul_32bit_exact(model, a, b, device='cpu'):
    """
    Test exactly like training: 16-bit × 16-bit → 32-bit.
    Uses same padding and evaluation as training.
    """
    bits = 32

    # Pad to 64 bits like training does
    a_bits = int_to_bits(a, bits)
    b_bits = int_to_bits(b, bits)
    a_bits = torch.nn.functional.pad(a_bits, (0, 64 - bits))
    b_bits = torch.nn.functional.pad(b_bits, (0, 64 - bits))

    # Run model - only look at first 32 bits (like training)
    with torch.no_grad():
        output = model(a_bits, b_bits)
        pred_bits = (torch.sigmoid(output[:, :bits]) > 0.5).float()

    return bits_to_int(pred_bits)


def add_64bit_simple(a, b):
    """Simple 64-bit addition (using Python for now)."""
    return a + b


def mul_64bit_composite(model, x, y, device='cpu'):
    """
    64-bit multiplication using model composition.

    x = A·2³² + B (A = high 32 bits, B = low 32 bits)
    y = C·2³² + D (C = high 32 bits, D = low 32 bits)

    x × y = AC·2⁶⁴ + (AD + BC)·2³² + BD

    IMPORTANT: This requires a model capable of 32-bit × 32-bit → 64-bit!
    - The 32-bit checkpoint only does 16-bit × 16-bit → 32-bit
    - The 64-bit checkpoint does 32-bit × 32-bit → 64-bit (what we need!)

    Since we need 128-bit result for exact answer, but we're doing 64-bit,
    we only keep the lower 64 bits (truncate AC·2⁶⁴).

    Result = ((AD + BC) << 32) + BD  (mod 2⁶⁴)
    """
    # Split into 32-bit halves
    A = (x >> 32).long()  # High 32 bits
    B = (x & 0xFFFFFFFF).long()  # Low 32 bits
    C = (y >> 32).long()
    D = (y & 0xFFFFFFFF).long()

    # 4 × 32-bit multiplications using neural model
    BD = mul_32bit(model, B, D, device)  # Low × Low
    AD = mul_32bit(model, A, D, device)  # High × Low
    BC = mul_32bit(model, B, C, device)  # Low × High
    # AC = mul_32bit(model, A, C, device)  # High × High (discarded for 64-bit)

    # Combine results (keeping only lower 64 bits)
    # Result = BD + ((AD + BC) << 32)
    middle = AD + BC  # This needs handling for overflow
    result = BD + ((middle & 0xFFFFFFFF) << 32)  # Take lower 32 bits of middle, shift

    # Handle middle overflow (add to high part)
    result = result & 0xFFFFFFFFFFFFFFFF  # Truncate to 64 bits

    return result


def test_composite_mul(model, num_tests=100, device='cpu'):
    """Test the composite 64-bit multiplication."""
    print(f"\n{'='*60}")
    print("Testing 64-bit MUL using 32-bit model composition")
    print(f"{'='*60}")

    correct = 0
    errors = []

    for i in range(num_tests):
        # Generate random 32-bit numbers (so result fits in 64 bits)
        x = torch.randint(0, 2**32, (1,), dtype=torch.long, device=device)
        y = torch.randint(0, 2**32, (1,), dtype=torch.long, device=device)

        expected = (x * y) & 0xFFFFFFFFFFFFFFFF  # True 64-bit result
        predicted = mul_64bit_composite(model, x, y, device)

        if predicted.item() == expected.item():
            correct += 1
        else:
            errors.append({
                'x': x.item(),
                'y': y.item(),
                'expected': expected.item(),
                'predicted': predicted.item()
            })
            if len(errors) <= 3:
                print(f"  Error: {x.item()} × {y.item()}")
                print(f"    Expected: {expected.item()}")
                print(f"    Got:      {predicted.item()}")

    accuracy = correct / num_tests * 100
    print(f"\nResults: {correct}/{num_tests} correct ({accuracy:.2f}%)")

    if errors:
        print(f"\nFirst few errors ({len(errors)} total):")
        for e in errors[:5]:
            print(f"  {e['x']:,} × {e['y']:,} = {e['expected']:,} (got {e['predicted']:,})")

    return accuracy


def test_32bit_direct(model, num_tests=100, device='cpu'):
    """
    Test the 32-bit model exactly like training does.
    Uses 16-bit × 16-bit → 32-bit to match training data.
    """
    print(f"\n{'='*60}")
    print("Testing 32-bit MUL (16-bit × 16-bit → 32-bit)")
    print(f"{'='*60}")

    correct = 0

    for i in range(num_tests):
        # Generate 16-bit numbers so result fits in 32 bits (SAME AS TRAINING)
        a = torch.randint(0, 2**16, (1,), dtype=torch.long, device=device)
        b = torch.randint(0, 2**16, (1,), dtype=torch.long, device=device)

        expected = (a * b) & 0xFFFFFFFF
        predicted = mul_32bit_exact(model, a, b, device)

        if (predicted & 0xFFFFFFFF).item() == expected.item():
            correct += 1

    accuracy = correct / num_tests * 100
    print(f"32-bit accuracy: {correct}/{num_tests} ({accuracy:.2f}%)")
    return accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to 32-bit MUL checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-tests', type=int, default=100)
    args = parser.parse_args()

    print(f"Loading model from: {args.checkpoint}")
    model = load_32bit_model(args.checkpoint, args.device)
    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test 32-bit directly first
    test_32bit_direct(model, args.num_tests, args.device)

    # Test 64-bit composite
    test_composite_mul(model, args.num_tests, args.device)
