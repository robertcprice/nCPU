#!/usr/bin/env python3
"""
TRULY NEURAL TIMER - LEARNED BEHAVIOR
======================================
The network LEARNS comparison logic from training examples.
NO pre-populated patterns. NO lookup tables.

The network learns:
- counter >= compare → should_fire (LEARNED from examples)
- enabled AND NOT masked → allow_irq (LEARNED from examples)

This is TRUE neural computation - learning the algorithm itself!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class TrulyNeuralTimerLearned(nn.Module):
    """
    Timer that LEARNS comparison logic - no hardcoded patterns.

    Architecture learns to compute:
    - Comparison: is counter >= compare?
    - Control logic: is enabled? is masked?
    - Output: should_fire, irq_signal
    """

    def __init__(self, input_bits=16, hidden_dim=128):
        super().__init__()
        self.input_bits = input_bits

        # Counter state IN WEIGHTS (current counter value)
        self.counter_state = nn.Parameter(torch.zeros(input_bits))

        # Compare value IN WEIGHTS
        self.compare_state = nn.Parameter(torch.zeros(input_bits))

        # Control state IN WEIGHTS [enabled, masked, status]
        self.control_state = nn.Parameter(torch.zeros(3))

        # LEARNED comparison network
        # Takes (counter_bits, compare_bits) and learns >= comparison
        self.comparison_net = nn.Sequential(
            nn.Linear(input_bits * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # 1 = counter >= compare
        )

        # LEARNED control logic network
        # Takes (comparison_result, control_bits) and learns fire/irq logic
        self.control_net = nn.Sequential(
            nn.Linear(1 + 3, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # [should_fire, irq_signal]
        )

        # LEARNED tick network (how counter advances)
        self.tick_net = nn.Sequential(
            nn.Linear(input_bits * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_bits),
        )

    def compare(self, counter_bits, compare_bits):
        """
        LEARN whether counter >= compare.

        This is the core neural computation - learning the >= operator!
        """
        compare_input = torch.cat([counter_bits, compare_bits], dim=-1)
        result = torch.sigmoid(self.comparison_net(compare_input))
        return result

    def check_fire(self, counter_bits, compare_bits, control_bits):
        """
        Check if timer should fire.

        Uses LEARNED comparison and control logic.
        """
        # Learn comparison
        comparison = self.compare(counter_bits, compare_bits)

        # Learn control logic
        control_input = torch.cat([comparison, control_bits], dim=-1)
        outputs = self.control_net(control_input)

        should_fire = torch.sigmoid(outputs[:, 0:1])
        irq_signal = torch.sigmoid(outputs[:, 1:2])

        return should_fire.squeeze(-1), irq_signal.squeeze(-1), comparison.squeeze(-1)

    def tick(self, counter_bits, tick_amount_bits):
        """
        LEARN how to advance counter by tick amount.
        """
        tick_input = torch.cat([counter_bits, tick_amount_bits], dim=-1)
        new_counter = torch.sigmoid(self.tick_net(tick_input))
        return new_counter


def int_to_bits(val, bits=16):
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def generate_batch(batch_size, device):
    """Generate training examples for comparison learning."""
    counters = []
    compares = []
    controls = []
    expected_ge = []  # counter >= compare
    expected_fire = []
    expected_irq = []

    for _ in range(batch_size):
        counter = random.randint(0, 65535)
        compare = random.randint(0, 65535)
        enabled = random.random() < 0.8
        masked = random.random() < 0.2

        # Ground truth: counter >= compare
        is_ge = float(counter >= compare)

        # Fire logic: (counter >= compare) AND enabled
        should_fire = is_ge * float(enabled)

        # IRQ logic: should_fire AND NOT masked
        irq = should_fire * (1.0 - float(masked))

        counters.append(int_to_bits(counter))
        compares.append(int_to_bits(compare))
        controls.append([float(enabled), float(masked), 0.0])
        expected_ge.append(is_ge)
        expected_fire.append(should_fire)
        expected_irq.append(irq)

    return (
        torch.stack(counters).to(device),
        torch.stack(compares).to(device),
        torch.tensor(controls, device=device),
        torch.tensor(expected_ge, device=device),
        torch.tensor(expected_fire, device=device),
        torch.tensor(expected_irq, device=device),
    )


def train():
    print("=" * 60)
    print("TRULY NEURAL TIMER - LEARNED BEHAVIOR")
    print("=" * 60)
    print(f"Device: {device}")
    print("Network LEARNS >= comparison from examples!")
    print("NO pre-populated patterns. NO lookup tables.")

    model = TrulyNeuralTimerLearned(
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
            counters, compares, controls, exp_ge, exp_fire, exp_irq = generate_batch(batch_size, device)

            optimizer.zero_grad()

            pred_fire, pred_irq, pred_ge = model.check_fire(counters, compares, controls)

            # Loss on all learned components
            loss_ge = F.binary_cross_entropy(pred_ge, exp_ge)
            loss_fire = F.binary_cross_entropy(pred_fire, exp_fire)
            loss_irq = F.binary_cross_entropy(pred_irq, exp_irq)

            loss = loss_ge + loss_fire + loss_irq
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        ge_correct = 0
        fire_correct = 0
        irq_correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(20):
                counters, compares, controls, exp_ge, exp_fire, exp_irq = generate_batch(256, device)
                pred_fire, pred_irq, pred_ge = model.check_fire(counters, compares, controls)

                ge_correct += ((pred_ge > 0.5).float() == exp_ge).sum().item()
                fire_correct += ((pred_fire > 0.5).float() == exp_fire).sum().item()
                irq_correct += ((pred_irq > 0.5).float() == exp_irq).sum().item()
                total += len(exp_ge)

        ge_acc = ge_correct / total
        fire_acc = fire_correct / total
        irq_acc = irq_correct / total
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} ge_acc={100*ge_acc:.1f}% fire={100*fire_acc:.1f}% irq={100*irq_acc:.1f}% [{elapsed:.1f}s]")

        if ge_acc > best_acc:
            best_acc = ge_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "ge_accuracy": ge_acc,
                "fire_accuracy": fire_acc,
                "irq_accuracy": irq_acc,
                "op_name": "TRULY_NEURAL_TIMER_LEARNED",
            }, "models/final/truly_neural_timer_learned_best.pt")
            print(f"  Saved (ge_acc={100*ge_acc:.1f}%)")

        if ge_acc >= 0.99 and fire_acc >= 0.99 and irq_acc >= 0.99:
            print("99%+ ACCURACY ON ALL!")
            break

    print(f"\nBest >= accuracy: {100*best_acc:.1f}%")

    # Final verification
    print("\nFinal verification - learned >= comparison:")
    model.eval()
    test_cases = [
        (100, 50, True, False),   # 100 >= 50, enabled, not masked → fire
        (50, 100, True, False),   # 50 < 100 → no fire
        (100, 100, True, False),  # 100 >= 100 → fire
        (100, 50, False, False),  # disabled → no fire
        (100, 50, True, True),    # masked → fire but no IRQ
    ]

    with torch.no_grad():
        for counter, compare, enabled, masked in test_cases:
            counter_bits = int_to_bits(counter).unsqueeze(0).to(device)
            compare_bits = int_to_bits(compare).unsqueeze(0).to(device)
            control_bits = torch.tensor([[float(enabled), float(masked), 0.0]], device=device)

            pred_fire, pred_irq, pred_ge = model.check_fire(counter_bits, compare_bits, control_bits)

            expected_ge = counter >= compare
            expected_fire = expected_ge and enabled
            expected_irq = expected_fire and not masked

            ge_ok = "✓" if (pred_ge.item() > 0.5) == expected_ge else "✗"
            fire_ok = "✓" if (pred_fire.item() > 0.5) == expected_fire else "✗"
            irq_ok = "✓" if (pred_irq.item() > 0.5) == expected_irq else "✗"

            print(f"  {counter}>={compare}? {ge_ok}  fire={fire_ok}  irq={irq_ok}")


if __name__ == "__main__":
    train()
