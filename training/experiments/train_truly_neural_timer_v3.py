#!/usr/bin/env python3
"""
TRULY NEURAL TIMER V3 - IMPROVED BOUNDARY LEARNING
====================================================
Fixes the equality edge case (100>=100) by:
1. Using separate networks for > and ==
2. Combining them with learned OR logic
3. Better training examples with balanced edge cases

The network learns:
- counter > compare (strict greater than)
- counter == compare (equality)
- counter >= compare = (>) OR (==)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class TrulyNeuralTimerV3(nn.Module):
    """
    Timer V3 - learns >= by decomposing into > and ==

    This handles the boundary condition better because:
    - Equality is learned separately (easier pattern)
    - Greater-than is learned separately (easier pattern)
    - Combining them is simpler than learning >= directly
    """

    def __init__(self, input_bits=16, hidden_dim=128):
        super().__init__()
        self.input_bits = input_bits

        # Counter and compare state IN WEIGHTS
        self.counter_state = nn.Parameter(torch.zeros(input_bits))
        self.compare_state = nn.Parameter(torch.zeros(input_bits))
        self.control_state = nn.Parameter(torch.zeros(3))

        # LEARNED greater-than network
        self.gt_net = nn.Sequential(
            nn.Linear(input_bits * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # counter > compare
        )

        # LEARNED equality network
        self.eq_net = nn.Sequential(
            nn.Linear(input_bits * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # counter == compare
        )

        # LEARNED combine network (learns OR logic)
        self.combine_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 1),  # >= = (>) OR (==)
        )

        # LEARNED control logic
        self.control_net = nn.Sequential(
            nn.Linear(1 + 3, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # [should_fire, irq_signal]
        )

    def compare_gt(self, counter_bits, compare_bits):
        """LEARN counter > compare."""
        compare_input = torch.cat([counter_bits, compare_bits], dim=-1)
        return torch.sigmoid(self.gt_net(compare_input))

    def compare_eq(self, counter_bits, compare_bits):
        """LEARN counter == compare."""
        compare_input = torch.cat([counter_bits, compare_bits], dim=-1)
        return torch.sigmoid(self.eq_net(compare_input))

    def compare_ge(self, counter_bits, compare_bits):
        """
        LEARN counter >= compare by combining > and ==.

        The network learns: (>) OR (==) = (>=)
        """
        gt_result = self.compare_gt(counter_bits, compare_bits)
        eq_result = self.compare_eq(counter_bits, compare_bits)

        # Combine with learned OR logic
        combined_input = torch.cat([gt_result, eq_result], dim=-1)
        ge_result = torch.sigmoid(self.combine_net(combined_input))

        return ge_result, gt_result, eq_result

    def check_fire(self, counter_bits, compare_bits, control_bits):
        """Check if timer should fire using decomposed comparison."""
        ge_result, gt_result, eq_result = self.compare_ge(counter_bits, compare_bits)

        # Control logic
        control_input = torch.cat([ge_result, control_bits], dim=-1)
        outputs = self.control_net(control_input)

        should_fire = torch.sigmoid(outputs[:, 0:1])
        irq_signal = torch.sigmoid(outputs[:, 1:2])

        return should_fire.squeeze(-1), irq_signal.squeeze(-1), ge_result.squeeze(-1), gt_result.squeeze(-1), eq_result.squeeze(-1)


def int_to_bits(val, bits=16):
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def generate_batch(batch_size, device):
    """Generate training examples with balanced edge cases."""
    counters = []
    compares = []
    controls = []
    expected_gt = []
    expected_eq = []
    expected_ge = []
    expected_fire = []
    expected_irq = []

    for i in range(batch_size):
        # Bias towards edge cases (same values)
        if random.random() < 0.25:
            # Equality case - IMPORTANT for learning boundary
            base = random.randint(0, 65535)
            counter = base
            compare = base
        elif random.random() < 0.25:
            # Near-equality case (counter = compare + 1)
            base = random.randint(0, 65534)
            counter = base + 1
            compare = base
        elif random.random() < 0.25:
            # Near-equality case (counter = compare - 1)
            base = random.randint(1, 65535)
            counter = base - 1
            compare = base
        else:
            # Random case
            counter = random.randint(0, 65535)
            compare = random.randint(0, 65535)

        enabled = random.random() < 0.8
        masked = random.random() < 0.2

        # Ground truth
        is_gt = float(counter > compare)
        is_eq = float(counter == compare)
        is_ge = float(counter >= compare)  # = is_gt OR is_eq

        should_fire = is_ge * float(enabled)
        irq = should_fire * (1.0 - float(masked))

        counters.append(int_to_bits(counter))
        compares.append(int_to_bits(compare))
        controls.append([float(enabled), float(masked), 0.0])
        expected_gt.append(is_gt)
        expected_eq.append(is_eq)
        expected_ge.append(is_ge)
        expected_fire.append(should_fire)
        expected_irq.append(irq)

    return (
        torch.stack(counters).to(device),
        torch.stack(compares).to(device),
        torch.tensor(controls, device=device),
        torch.tensor(expected_gt, device=device),
        torch.tensor(expected_eq, device=device),
        torch.tensor(expected_ge, device=device),
        torch.tensor(expected_fire, device=device),
        torch.tensor(expected_irq, device=device),
    )


def train():
    print("=" * 60)
    print("TRULY NEURAL TIMER V3 - DECOMPOSED COMPARISON")
    print("=" * 60)
    print(f"Device: {device}")
    print("Learns >= by decomposing into > and ==")
    print("Better edge case handling!")

    model = TrulyNeuralTimerV3(
        input_bits=16,
        hidden_dim=128
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 512

    for epoch in range(200):
        model.train()
        total_loss = 0
        t0 = time.time()

        for _ in range(100):
            counters, compares, controls, exp_gt, exp_eq, exp_ge, exp_fire, exp_irq = generate_batch(batch_size, device)

            optimizer.zero_grad()

            pred_fire, pred_irq, pred_ge, pred_gt, pred_eq = model.check_fire(counters, compares, controls)

            # Losses on all components
            loss_gt = F.binary_cross_entropy(pred_gt.squeeze(-1), exp_gt)
            loss_eq = F.binary_cross_entropy(pred_eq.squeeze(-1), exp_eq)
            loss_ge = F.binary_cross_entropy(pred_ge, exp_ge)
            loss_fire = F.binary_cross_entropy(pred_fire, exp_fire)
            loss_irq = F.binary_cross_entropy(pred_irq, exp_irq)

            # Weight equality higher since it's the edge case
            loss = loss_gt + 2.0 * loss_eq + loss_ge + loss_fire + loss_irq
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        gt_correct = 0
        eq_correct = 0
        ge_correct = 0
        fire_correct = 0
        irq_correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(20):
                counters, compares, controls, exp_gt, exp_eq, exp_ge, exp_fire, exp_irq = generate_batch(256, device)
                pred_fire, pred_irq, pred_ge, pred_gt, pred_eq = model.check_fire(counters, compares, controls)

                gt_correct += ((pred_gt.squeeze(-1) > 0.5).float() == exp_gt).sum().item()
                eq_correct += ((pred_eq.squeeze(-1) > 0.5).float() == exp_eq).sum().item()
                ge_correct += ((pred_ge > 0.5).float() == exp_ge).sum().item()
                fire_correct += ((pred_fire > 0.5).float() == exp_fire).sum().item()
                irq_correct += ((pred_irq > 0.5).float() == exp_irq).sum().item()
                total += len(exp_ge)

        gt_acc = gt_correct / total
        eq_acc = eq_correct / total
        ge_acc = ge_correct / total
        fire_acc = fire_correct / total
        irq_acc = irq_correct / total
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} gt={100*gt_acc:.1f}% eq={100*eq_acc:.1f}% ge={100*ge_acc:.1f}% fire={100*fire_acc:.1f}% irq={100*irq_acc:.1f}% [{elapsed:.1f}s]")

        if ge_acc > best_acc:
            best_acc = ge_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "gt_accuracy": gt_acc,
                "eq_accuracy": eq_acc,
                "ge_accuracy": ge_acc,
                "fire_accuracy": fire_acc,
                "irq_accuracy": irq_acc,
                "op_name": "TRULY_NEURAL_TIMER_V3",
            }, "models/final/truly_neural_timer_v3_best.pt")
            print(f"  Saved (ge_acc={100*ge_acc:.1f}%)")

        if ge_acc >= 0.99 and fire_acc >= 0.99 and irq_acc >= 0.99:
            print("99%+ ACCURACY ON ALL!")
            break

    print(f"\nBest >= accuracy: {100*best_acc:.1f}%")

    # Final verification
    print("\nFinal verification - decomposed comparison:")
    model.eval()
    test_cases = [
        (100, 50, True, False),   # 100 > 50, enabled, not masked → fire
        (50, 100, True, False),   # 50 < 100 → no fire
        (100, 100, True, False),  # 100 == 100 → fire (EDGE CASE)
        (100, 99, True, False),   # 100 > 99 → fire
        (99, 100, True, False),   # 99 < 100 → no fire
        (100, 50, False, False),  # disabled → no fire
        (100, 50, True, True),    # masked → fire but no IRQ
        (0, 0, True, False),      # 0 == 0 → fire (EDGE CASE)
        (65535, 65535, True, False),  # max == max → fire (EDGE CASE)
    ]

    with torch.no_grad():
        all_pass = True
        for counter, compare, enabled, masked in test_cases:
            counter_bits = int_to_bits(counter).unsqueeze(0).to(device)
            compare_bits = int_to_bits(compare).unsqueeze(0).to(device)
            control_bits = torch.tensor([[float(enabled), float(masked), 0.0]], device=device)

            pred_fire, pred_irq, pred_ge, pred_gt, pred_eq = model.check_fire(counter_bits, compare_bits, control_bits)

            expected_gt = counter > compare
            expected_eq = counter == compare
            expected_ge = counter >= compare
            expected_fire = expected_ge and enabled
            expected_irq = expected_fire and not masked

            gt_ok = (pred_gt.item() > 0.5) == expected_gt
            eq_ok = (pred_eq.item() > 0.5) == expected_eq
            ge_ok = (pred_ge.item() > 0.5) == expected_ge
            fire_ok = (pred_fire.item() > 0.5) == expected_fire
            irq_ok = (pred_irq.item() > 0.5) == expected_irq

            gt_sym = "✓" if gt_ok else "✗"
            eq_sym = "✓" if eq_ok else "✗"
            ge_sym = "✓" if ge_ok else "✗"
            fire_sym = "✓" if fire_ok else "✗"
            irq_sym = "✓" if irq_ok else "✗"

            if not all([gt_ok, eq_ok, ge_ok, fire_ok, irq_ok]):
                all_pass = False

            print(f"  {counter}>={compare}? gt={gt_sym} eq={eq_sym} ge={ge_sym} fire={fire_sym} irq={irq_sym}")

        if all_pass:
            print("\n✓ ALL VERIFICATION TESTS PASSED!")
        else:
            print("\n✗ Some verification tests failed")


if __name__ == "__main__":
    train()
