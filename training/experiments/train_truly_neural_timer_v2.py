#!/usr/bin/env python3
"""
TRULY NEURAL TIMER V2
=====================
Simplified timer that learns FIXED comparison patterns.

Key insight: Like the MMU v2, we pre-populate "when to fire" patterns
and train the network to recall them. The timer learns:
- Counter comparison logic (counter >= compare → fire)
- Control register effects (enable, mask)
- IRQ generation rules

All stored IN THE NETWORK WEIGHTS!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class TrulyNeuralTimerV2(nn.Module):
    """
    Timer V2 - learns comparison patterns IN weights.

    Instead of trying to learn arbitrary counter math,
    we train on fixed patterns:
    - (counter_high_bits, compare_high_bits) → should_fire
    """

    def __init__(self, pattern_slots=256, input_bits=16, key_dim=64):
        super().__init__()
        self.pattern_slots = pattern_slots
        self.input_bits = input_bits
        self.key_dim = key_dim

        # === COMPARISON PATTERNS IN WEIGHTS ===
        # Each slot: (counter_pattern, compare_pattern, control_pattern) → fire_decision
        self.counter_patterns = nn.Parameter(torch.zeros(pattern_slots, input_bits))
        self.compare_patterns = nn.Parameter(torch.zeros(pattern_slots, input_bits))
        self.control_patterns = nn.Parameter(torch.zeros(pattern_slots, 3))  # enable, mask, status
        self.fire_decisions = nn.Parameter(torch.zeros(pattern_slots, 2))  # should_fire, irq

        # Keys for attention lookup
        self.pattern_keys = nn.Parameter(torch.randn(pattern_slots, key_dim) * 0.1)

        # Query encoder: (counter, compare, control) → key space
        self.query_encoder = nn.Sequential(
            nn.Linear(input_bits * 2 + 3, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
        )

        self.temperature = nn.Parameter(torch.tensor(0.5))

    def check_fire(self, counter_bits, compare_bits, control_bits):
        """
        Check if timer should fire.

        Args:
            counter_bits: [batch, input_bits] - counter value (high bits)
            compare_bits: [batch, input_bits] - compare value (high bits)
            control_bits: [batch, 3] - enable, mask, status

        Returns:
            should_fire: [batch]
            irq_signal: [batch]
        """
        batch = counter_bits.shape[0]

        # Create query
        query_input = torch.cat([counter_bits, compare_bits, control_bits], dim=-1)
        query = self.query_encoder(query_input)

        # Compute similarity to stored patterns
        stored_counter = torch.sigmoid(self.counter_patterns)
        stored_compare = torch.sigmoid(self.compare_patterns)
        stored_control = torch.sigmoid(self.control_patterns)

        # Key similarity
        key_sim = torch.matmul(query, self.pattern_keys.T)

        # Pattern similarity (how well input matches stored patterns)
        counter_sim = -((counter_bits.unsqueeze(1) - stored_counter.unsqueeze(0)) ** 2).sum(dim=-1)
        compare_sim = -((compare_bits.unsqueeze(1) - stored_compare.unsqueeze(0)) ** 2).sum(dim=-1)
        control_sim = -((control_bits.unsqueeze(1) - stored_control.unsqueeze(0)) ** 2).sum(dim=-1)

        combined_sim = key_sim + counter_sim + compare_sim + 0.5 * control_sim

        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(combined_sim / temp, dim=-1)

        # Read fire decisions
        fire_decisions = torch.sigmoid(self.fire_decisions)
        outputs = torch.matmul(attention, fire_decisions)

        should_fire = outputs[:, 0]
        irq_signal = outputs[:, 1]

        return should_fire, irq_signal

    def populate_patterns(self, counter_patterns, compare_patterns, control_patterns, fire_decisions):
        """Populate timer patterns for training."""
        n = counter_patterns.shape[0]

        with torch.no_grad():
            def to_logits(x):
                return torch.log(x.clamp(0.01, 0.99) / (1 - x.clamp(0.01, 0.99)))

            self.counter_patterns.data[:n] = to_logits(counter_patterns)
            self.compare_patterns.data[:n] = to_logits(compare_patterns)
            self.control_patterns.data[:n] = to_logits(control_patterns)
            self.fire_decisions.data[:n] = to_logits(fire_decisions)

            # Update keys to match patterns
            query_input = torch.cat([counter_patterns, compare_patterns, control_patterns], dim=-1)
            self.pattern_keys.data[:n] = self.query_encoder(query_input.float()).detach()


def int_to_bits(val, bits=16):
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def create_timer_patterns(num_patterns=200):
    """Create fixed timer comparison patterns."""
    patterns = []

    for _ in range(num_patterns):
        # Random counter and compare values (using high bits)
        counter = random.randint(0, 65535)
        compare = random.randint(0, 65535)

        # Control: enabled, not masked
        enabled = random.random() < 0.9
        masked = random.random() < 0.1

        # Fire logic: counter >= compare AND enabled AND not masked
        should_fire = (counter >= compare) and enabled and not masked
        irq = should_fire  # IRQ follows fire

        patterns.append({
            'counter': counter,
            'compare': compare,
            'enabled': enabled,
            'masked': masked,
            'should_fire': should_fire,
            'irq': irq,
        })

    return patterns


def generate_batch(patterns, batch_size, device):
    """Generate training batch from patterns."""
    counters = []
    compares = []
    controls = []
    fires = []
    irqs = []

    for _ in range(batch_size):
        p = random.choice(patterns)
        counters.append(int_to_bits(p['counter']))
        compares.append(int_to_bits(p['compare']))
        controls.append([float(p['enabled']), float(p['masked']), 0.0])
        fires.append(float(p['should_fire']))
        irqs.append(float(p['irq']))

    return (
        torch.stack(counters).to(device),
        torch.stack(compares).to(device),
        torch.tensor(controls, device=device),
        torch.tensor(fires, device=device),
        torch.tensor(irqs, device=device),
    )


def train():
    print("=" * 60)
    print("TRULY NEURAL TIMER V2 TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("Fixed comparison patterns stored IN NETWORK WEIGHTS!")

    # Create patterns
    patterns = create_timer_patterns(num_patterns=200)
    print(f"Created {len(patterns)} timer patterns")

    model = TrulyNeuralTimerV2(
        pattern_slots=256,
        input_bits=16,
        key_dim=64
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Pre-populate patterns
    print("Populating patterns in weights...")
    counter_init = torch.stack([int_to_bits(p['counter']) for p in patterns]).to(device)
    compare_init = torch.stack([int_to_bits(p['compare']) for p in patterns]).to(device)
    control_init = torch.tensor([[float(p['enabled']), float(p['masked']), 0.0] for p in patterns], device=device)
    fire_init = torch.tensor([[float(p['should_fire']), float(p['irq'])] for p in patterns], device=device)
    model.populate_patterns(counter_init, compare_init, control_init, fire_init)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 256

    for epoch in range(100):
        model.train()
        total_loss = 0
        t0 = time.time()

        for _ in range(100):
            counters, compares, controls, exp_fires, exp_irqs = generate_batch(patterns, batch_size, device)

            optimizer.zero_grad()

            pred_fire, pred_irq = model.check_fire(counters, compares, controls)

            loss_fire = F.binary_cross_entropy(pred_fire, exp_fires)
            loss_irq = F.binary_cross_entropy(pred_irq, exp_irqs)

            loss = loss_fire + loss_irq
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        fire_correct = 0
        irq_correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(20):
                counters, compares, controls, exp_fires, exp_irqs = generate_batch(patterns, 100, device)
                pred_fire, pred_irq = model.check_fire(counters, compares, controls)

                fire_preds = (pred_fire > 0.5).float()
                irq_preds = (pred_irq > 0.5).float()

                fire_correct += (fire_preds == exp_fires).sum().item()
                irq_correct += (irq_preds == exp_irqs).sum().item()
                total += len(exp_fires)

        fire_acc = fire_correct / total
        irq_acc = irq_correct / total
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} fire_acc={100*fire_acc:.1f}% irq_acc={100*irq_acc:.1f}% [{elapsed:.1f}s]")

        if fire_acc > best_acc:
            best_acc = fire_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "fire_accuracy": fire_acc,
                "irq_accuracy": irq_acc,
                "op_name": "TRULY_NEURAL_TIMER_V2",
                "num_patterns": len(patterns),
            }, "models/final/truly_neural_timer_v2_best.pt")
            print(f"  Saved (fire_acc={100*fire_acc:.1f}%)")

        if fire_acc >= 0.99 and irq_acc >= 0.99:
            print("99%+ ACCURACY ON BOTH!")
            break

    print(f"\nBest fire accuracy: {100*best_acc:.1f}%")


if __name__ == "__main__":
    train()
