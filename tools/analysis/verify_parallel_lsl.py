#!/usr/bin/env python3
"""Quick verification of the parallel neural LSL model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"


class ParallelDecomposedLSL(nn.Module):
    """Parallelized version of DecomposedShiftNet - NO hardcoded matrices."""

    def __init__(self, max_bits=64, hidden_dim=512):
        super().__init__()
        self.max_bits = max_bits

        self.shift_decoder = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)
        )

        self.index_net = nn.Sequential(
            nn.Linear(max_bits * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)
        )

        self.validity_net = nn.Sequential(
            nn.Linear(max_bits * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.register_buffer('pos_onehots', torch.eye(max_bits))
        self.register_buffer('temp', torch.tensor(1.0))

    def set_temperature(self, t):
        self.temp.fill_(t)

    def forward(self, a_bits, shift_bits, return_aux=False):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]

        shift_logits = self.shift_decoder(shift_bits)[:, :bits]

        if self.training:
            shift_soft = F.softmax(shift_logits / self.temp, dim=-1)
        else:
            shift_soft = F.one_hot(shift_logits.argmax(dim=-1), bits).float()

        pos_inputs = self.pos_onehots[:bits, :bits]
        pos_inputs = pos_inputs.unsqueeze(0).expand(batch, -1, -1)
        shift_expanded = shift_soft.unsqueeze(1).expand(-1, bits, -1)
        combined = torch.cat([pos_inputs, shift_expanded], dim=-1)

        combined_flat = combined.view(batch * bits, bits * 2)
        index_logits_flat = self.index_net(combined_flat)[:, :bits]
        index_logits = index_logits_flat.view(batch, bits, bits)

        if self.training:
            index_soft = F.softmax(index_logits / self.temp, dim=-1)
        else:
            index_soft = F.one_hot(index_logits.argmax(dim=-1), bits).float()

        gathered = torch.bmm(index_soft, a_bits.unsqueeze(-1)).squeeze(-1)

        validity_logits_flat = self.validity_net(combined_flat)
        validity_logits = validity_logits_flat.view(batch, bits)

        output = gathered * torch.sigmoid(validity_logits)

        if return_aux:
            return output, shift_logits, index_logits, validity_logits
        return output


def int_to_bits(val, bits=64):
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def bits_to_int(bits_t):
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.cpu().tolist()))


def main():
    print("=" * 70)
    print("PARALLEL NEURAL LSL - Verification")
    print("=" * 70)
    print(f"Device: {device}")

    model = ParallelDecomposedLSL(max_bits=64, hidden_dim=768).to(device)

    model_path = "models/parallel_neural/LSL_parallel_shift63_100pct.pt"
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded: {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    model.eval()
    model.set_temperature(0.01)

    # Edge case tests
    print("\n" + "=" * 70)
    print("EDGE CASE TESTS")
    print("=" * 70)

    test_cases = [
        (1, 0, "Identity"),
        (1, 1, "Simple shift"),
        (1, 63, "Max shift single bit"),
        (0xFFFFFFFFFFFFFFFF, 0, "All 1s, no shift"),
        (0xFFFFFFFFFFFFFFFF, 1, "All 1s, shift 1"),
        (0xAAAAAAAAAAAAAAAA, 32, "Alternating, shift 32"),
        (0x8000000000000000, 0, "MSB only"),
        (1, 4, "1 << 4"),
        (255, 8, "255 << 8"),
        (0xF0F0F0F0F0F0F0F0, 4, "Pattern << 4"),
        (0x1234567890ABCDEF, 8, "Complex << 8"),
        (0x0000000000000001, 63, "1 << 63 (hardest)"),
    ]

    passed = 0
    with torch.no_grad():
        for val, shift, desc in test_cases:
            input_bits = int_to_bits(val, 64).unsqueeze(0).to(device)
            shift_bits = int_to_bits(shift, 64).unsqueeze(0).to(device)
            output = model(input_bits, shift_bits)
            result = bits_to_int(output[0])
            expected = (val << shift) & ((1 << 64) - 1)
            ok = result == expected
            status = "OK" if ok else f"FAIL (got {hex(result)})"
            print(f"  {desc:30s}: {hex(val)} << {shift:2d} = {status}")
            if ok:
                passed += 1

    print(f"\nEdge cases: {passed}/{len(test_cases)}")

    # Random tests
    print("\n" + "=" * 70)
    print("RANDOM TESTS (1000 samples)")
    print("=" * 70)

    random_passed = 0
    with torch.no_grad():
        for _ in range(1000):
            val = random.randint(0, (1 << 64) - 1)
            shift = random.randint(0, 63)
            expected = (val << shift) & ((1 << 64) - 1)

            input_bits = int_to_bits(val, 64).unsqueeze(0).to(device)
            shift_bits = int_to_bits(shift, 64).unsqueeze(0).to(device)
            output = model(input_bits, shift_bits)
            result = bits_to_int(output[0])

            if result == expected:
                random_passed += 1

    print(f"Random tests: {random_passed}/1000")

    # Final verdict
    print("\n" + "=" * 70)
    total_passed = passed + random_passed
    total_tests = len(test_cases) + 1000
    if total_passed == total_tests:
        print(f">>> ALL {total_tests} TESTS PASSED! <<<")
    else:
        print(f">>> {total_passed}/{total_tests} TESTS PASSED <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
