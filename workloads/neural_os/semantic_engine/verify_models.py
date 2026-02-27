#!/usr/bin/env python3
"""
Verify existing Neural CPU models for 100% accuracy.

Tests stack, pointer, and other models that exist but haven't been validated.
"""

import torch
import torch.nn as nn
import sys
import os
import argparse

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bits_to_int(bits):
    """Convert bit tensor to integer using int64."""
    num_bits = min(bits.shape[-1], 64)
    result = torch.zeros(bits.shape[0], dtype=torch.int64, device=bits.device)
    for i in range(num_bits):
        result = result + (bits[:, i].long() << i)
    return result


def int_to_bits(x, bits=64):
    """Convert integer to bit tensor."""
    return ((x.unsqueeze(-1) >> torch.arange(bits, device=x.device)) & 1).float()


def test_shift_model(model_path, op='LSL', bits=64, num_tests=1000, device='cpu'):
    """Test shift operation model."""
    from train_shift_fast import ShiftNet

    print(f"\nTesting {op} model: {os.path.basename(model_path)}")

    try:
        model = ShiftNet(bits=bits, d_model=256, nhead=8, num_layers=4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model = model.to(device)
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return 0.0

    correct = 0
    for _ in range(num_tests):
        # Generate test data
        a_bits = torch.randint(0, 2, (1, bits), device=device).float()
        shift = torch.randint(0, bits, (1,), device=device)
        shift_bits = ((shift.unsqueeze(1) >> torch.arange(6, device=device)) & 1).float()

        # Compute expected result
        positions = torch.arange(bits, device=device).unsqueeze(0)
        if op == 'LSL':
            src_pos = positions - shift.unsqueeze(1)
            valid = src_pos >= 0
        elif op == 'LSR':
            src_pos = positions + shift.unsqueeze(1)
            valid = src_pos < bits
        elif op == 'ASR':
            src_pos = positions + shift.unsqueeze(1)
            valid = src_pos < bits
            # ASR fills with sign bit
            sign_bit = a_bits[:, bits-1:bits]
            # For invalid positions, use sign bit
        elif op == 'ROL':
            src_pos = (positions - shift.unsqueeze(1)) % bits
            valid = torch.ones_like(src_pos, dtype=torch.bool)
        elif op == 'ROR':
            src_pos = (positions + shift.unsqueeze(1)) % bits
            valid = torch.ones_like(src_pos, dtype=torch.bool)

        src_pos = src_pos.clamp(0, bits-1)
        expected = torch.gather(a_bits, 1, src_pos) * valid.float()

        # For ASR, fill invalid positions with sign bit
        if op == 'ASR':
            sign_fill = sign_bit.expand(-1, bits) * (~valid).float()
            expected = expected + sign_fill

        # Get model prediction
        with torch.no_grad():
            output = model(a_bits, shift_bits)
            pred = (output > 0.5).float()

        if (pred == expected).all():
            correct += 1

    accuracy = correct / num_tests * 100
    status = "✅" if accuracy >= 100.0 else "⚠️"
    print(f"  {status} Accuracy: {correct}/{num_tests} ({accuracy:.2f}%)")
    return accuracy


def test_generic_model(model_path, model_class, test_fn, name, num_tests=1000, device='cpu'):
    """Generic model tester."""
    print(f"\nTesting {name}: {os.path.basename(model_path)}")

    try:
        # Try to load and analyze the checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict):
            # It's a state dict or checkpoint with metadata
            keys = list(checkpoint.keys())[:10]
            print(f"  Checkpoint keys (first 10): {keys}")

            # Try to infer model structure from state dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Analyze layer shapes
            for key in list(state_dict.keys())[:5]:
                if hasattr(state_dict[key], 'shape'):
                    print(f"    {key}: {state_dict[key].shape}")
        else:
            print(f"  ⚠️ Checkpoint is not a dict: {type(checkpoint)}")

        return 0.0  # Need model class to test properly

    except Exception as e:
        print(f"  ❌ Failed to analyze: {e}")
        return 0.0


def verify_mul_model(checkpoint_path, bits=64, num_tests=1000, device='cpu'):
    """Verify the MUL model we just trained."""
    from train_mul_add_hybrid import PureParallelMulWithADDv2, generate_mul_data

    print(f"\nVerifying MUL model: {os.path.basename(checkpoint_path)}")

    try:
        model = PureParallelMulWithADDv2(max_bits=bits, d_model=384, nhead=16, num_layers=7)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        model = model.to(device)
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return 0.0

    # Generate test data
    a_bits, b_bits, target = generate_mul_data(num_tests, bits, device)

    with torch.no_grad():
        output = model(a_bits, b_bits)
        pred = (torch.sigmoid(output[:, :bits]) > 0.5).float()

    correct = (pred == target).all(dim=1).float().sum().item()
    accuracy = correct / num_tests * 100

    status = "✅" if accuracy >= 100.0 else "⚠️"
    print(f"  {status} Accuracy: {int(correct)}/{num_tests} ({accuracy:.2f}%)")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-tests', type=int, default=1000)
    args = parser.parse_args()

    print("=" * 70)
    print("NEURAL CPU MODEL VERIFICATION")
    print("=" * 70)

    results = {}

    # 1. Verify MUL model (our new breakthrough)
    mul_path = "models/MUL_64bit_add_hybrid_v3.5_100pct.pt"
    if os.path.exists(mul_path):
        results['MUL'] = verify_mul_model(mul_path, 64, args.num_tests, args.device)

    # 2. Check 100pct models directory
    models_100pct = "../models_100pct"
    if os.path.exists(models_100pct):
        print(f"\n{'='*70}")
        print("100% ACCURACY MODELS")
        print("="*70)

        for f in os.listdir(models_100pct):
            if f.endswith('.pt'):
                path = os.path.join(models_100pct, f)
                name = f.replace('_64bit_100pct.pt', '').replace('_100pct.pt', '')

                if 'LSL' in f or 'LSR' in f:
                    op = 'LSL' if 'LSL' in f else 'LSR'
                    results[name] = test_shift_model(path, op, 64, args.num_tests, args.device)
                else:
                    # Analyze the checkpoint structure
                    results[name] = test_generic_model(path, None, None, name, args.num_tests, args.device)

    # 3. Check trained_models/64bit
    trained_64bit = "trained_models/64bit"
    if os.path.exists(trained_64bit):
        print(f"\n{'='*70}")
        print("TRAINED 64-BIT MODELS (Not 100% verified)")
        print("="*70)

        for f in os.listdir(trained_64bit):
            if f.endswith('.pt'):
                path = os.path.join(trained_64bit, f)
                name = f.replace('kvrm64.pt', '').replace('.pt', '')
                results[f'trained_{name}'] = test_generic_model(path, None, None, name, args.num_tests, args.device)

    # Summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print("="*70)

    for name, acc in sorted(results.items()):
        if acc >= 100.0:
            print(f"  ✅ {name}: {acc:.2f}%")
        elif acc >= 95.0:
            print(f"  🔶 {name}: {acc:.2f}% (close!)")
        else:
            print(f"  ⚠️ {name}: {acc:.2f}% (needs work)")


if __name__ == '__main__':
    main()
