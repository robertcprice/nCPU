#!/usr/bin/env python3
"""
TRULY NEURAL TIMER
==================
System timer with ALL state stored IN THE NETWORK WEIGHTS.

Like the Truly Neural Register File:
- Counter value stored as nn.Parameter
- Compare value stored as nn.Parameter
- Control register stored as nn.Parameter
- Tick/fire logic is LEARNED, not hardcoded!

The timer learns:
1. How to increment counter on tick
2. When to fire based on counter vs compare
3. How to generate interrupt signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class TrulyNeuralTimer(nn.Module):
    """
    System timer where ALL state is stored IN the neural network weights.

    Architecture:
    - counter_state: nn.Parameter [64] - 64-bit counter IN weights
    - compare_value: nn.Parameter [64] - compare register IN weights
    - control_state: nn.Parameter [8] - control bits IN weights
    - tick_transform: learned counter increment
    - fire_detector: learned comparison logic
    """

    def __init__(self, counter_bits=64, key_dim=64):
        super().__init__()
        self.counter_bits = counter_bits
        self.key_dim = key_dim

        # === ALL STATE IN WEIGHTS ===
        # Counter (64-bit as individual bit parameters)
        self.counter_state = nn.Parameter(torch.zeros(counter_bits))

        # Compare value (what counter is compared against)
        self.compare_value = nn.Parameter(torch.zeros(counter_bits))

        # Control register: [enable, mask, status, ...]
        self.control_state = nn.Parameter(torch.zeros(8))

        # === LEARNED OPERATIONS ===
        # Tick encoder: current counter → increment decision
        self.tick_encoder = nn.Sequential(
            nn.Linear(counter_bits + 8, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, counter_bits),  # New counter value
        )

        # Fire detector: (counter, compare, control) → should_fire
        self.fire_detector = nn.Sequential(
            nn.Linear(counter_bits * 2 + 8, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, key_dim // 2),
            nn.GELU(),
            nn.Linear(key_dim // 2, 2),  # [should_fire, irq_pending]
        )

        # Update learning rate
        self.update_lr = nn.Parameter(torch.tensor(0.5))

    def tick(self, num_ticks_bits):
        """
        Advance timer by specified ticks.

        Args:
            num_ticks_bits: [batch, counter_bits] - number of ticks as bits

        Returns:
            new_counter: [batch, counter_bits]
            fired: [batch] - whether timer fired
            irq: [batch] - whether to raise interrupt
        """
        batch = num_ticks_bits.shape[0]

        # Get current state
        counter = torch.sigmoid(self.counter_state).unsqueeze(0).expand(batch, -1)
        control = torch.sigmoid(self.control_state).unsqueeze(0).expand(batch, -1)

        # Compute new counter via learned transformation
        tick_input = torch.cat([counter + num_ticks_bits, control], dim=-1)
        counter_delta = self.tick_encoder(tick_input)

        # New counter value (with overflow handling learned)
        new_counter = torch.sigmoid(counter + counter_delta)

        # Update counter state (Hebbian-style)
        lr = torch.clamp(self.update_lr.abs(), 0.01, 1.0)
        new_counter_mean = new_counter.mean(dim=0)
        counter_logits = torch.log(new_counter_mean.clamp(0.01, 0.99) / (1 - new_counter_mean.clamp(0.01, 0.99)))
        self.counter_state.data = (1 - lr) * self.counter_state.data + lr * counter_logits

        # Check if timer should fire
        compare = torch.sigmoid(self.compare_value).unsqueeze(0).expand(batch, -1)
        fire_input = torch.cat([new_counter, compare, control], dim=-1)
        fire_output = self.fire_detector(fire_input)

        fired = torch.sigmoid(fire_output[:, 0])
        irq = torch.sigmoid(fire_output[:, 1])

        # Only fire if enabled (control[0]) and not masked (control[1])
        enabled = control[:, 0]
        masked = control[:, 1]
        actual_irq = irq * enabled * (1 - masked)

        return new_counter, fired, actual_irq

    def read_counter(self):
        """Read current counter value."""
        return torch.sigmoid(self.counter_state)

    def write_compare(self, compare_bits):
        """Write compare value."""
        batch = compare_bits.shape[0]
        lr = torch.clamp(self.update_lr.abs(), 0.01, 1.0)
        compare_mean = compare_bits.mean(dim=0)
        compare_logits = torch.log(compare_mean.clamp(0.01, 0.99) / (1 - compare_mean.clamp(0.01, 0.99)))
        self.compare_value.data = (1 - lr) * self.compare_value.data + lr * compare_logits

        # Clear status bit
        self.control_state.data[2] = self.control_state.data[2] * 0.5

    def write_control(self, control_bits):
        """Write control register."""
        batch = control_bits.shape[0]
        lr = torch.clamp(self.update_lr.abs(), 0.01, 1.0)
        control_mean = control_bits.mean(dim=0)
        control_logits = torch.log(control_mean.clamp(0.01, 0.99) / (1 - control_mean.clamp(0.01, 0.99)))
        self.control_state.data = (1 - lr) * self.control_state.data + lr * control_logits


def int_to_bits(val, bits=64):
    """Convert integer to bit tensor."""
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def bits_to_int(bits_t):
    """Convert bit tensor to integer."""
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.tolist()))


def generate_batch(batch_size, device):
    """Generate training batch for timer operations."""
    counters = []
    ticks = []
    compares = []
    controls = []
    expected_fires = []
    expected_irqs = []

    for _ in range(batch_size):
        # Random counter value
        counter_val = random.randint(0, 2**32 - 1)
        # Random tick amount
        tick_val = random.randint(1, 1000)
        # Random compare value (sometimes before, sometimes after counter+tick)
        if random.random() < 0.5:
            # Will fire
            compare_val = counter_val + random.randint(0, tick_val)
            should_fire = 1.0
        else:
            # Won't fire
            compare_val = counter_val + tick_val + random.randint(1, 10000)
            should_fire = 0.0

        # Control: [enable, mask, status, ...]
        enabled = random.random() < 0.8
        masked = random.random() < 0.2

        counters.append(int_to_bits(counter_val))
        ticks.append(int_to_bits(tick_val))
        compares.append(int_to_bits(compare_val))
        controls.append([float(enabled), float(masked), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        expected_fires.append(should_fire)
        # IRQ only if enabled, not masked, and should fire
        expected_irqs.append(should_fire * float(enabled) * (1 - float(masked)))

    return (
        torch.stack(counters).to(device),
        torch.stack(ticks).to(device),
        torch.stack(compares).to(device),
        torch.tensor(controls, device=device),
        torch.tensor(expected_fires, device=device),
        torch.tensor(expected_irqs, device=device),
    )


def train():
    print("=" * 60)
    print("TRULY NEURAL TIMER TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("ALL timer state stored IN NETWORK WEIGHTS!")

    model = TrulyNeuralTimer(
        counter_bits=64,
        key_dim=64
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 256

    for epoch in range(100):
        model.train()
        total_loss = 0
        t0 = time.time()

        for _ in range(100):
            counters, ticks, compares, controls, exp_fires, exp_irqs = generate_batch(batch_size, device)

            # Initialize timer state for this batch
            model.counter_state.data = torch.log(counters.mean(dim=0).clamp(0.01, 0.99) / (1 - counters.mean(dim=0).clamp(0.01, 0.99)))
            model.compare_value.data = torch.log(compares.mean(dim=0).clamp(0.01, 0.99) / (1 - compares.mean(dim=0).clamp(0.01, 0.99)))
            model.control_state.data = torch.log(controls.mean(dim=0).clamp(0.01, 0.99) / (1 - controls.mean(dim=0).clamp(0.01, 0.99)))

            optimizer.zero_grad()

            # Tick
            new_counter, fired, irq = model.tick(ticks)

            # Losses
            loss_fire = F.binary_cross_entropy(fired, exp_fires)
            loss_irq = F.binary_cross_entropy(irq, exp_irqs)

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
                counters, ticks, compares, controls, exp_fires, exp_irqs = generate_batch(100, device)

                model.counter_state.data = torch.log(counters.mean(dim=0).clamp(0.01, 0.99) / (1 - counters.mean(dim=0).clamp(0.01, 0.99)))
                model.compare_value.data = torch.log(compares.mean(dim=0).clamp(0.01, 0.99) / (1 - compares.mean(dim=0).clamp(0.01, 0.99)))
                model.control_state.data = torch.log(controls.mean(dim=0).clamp(0.01, 0.99) / (1 - controls.mean(dim=0).clamp(0.01, 0.99)))

                new_counter, fired, irq = model.tick(ticks)

                fire_preds = (fired > 0.5).float()
                irq_preds = (irq > 0.5).float()

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
                "op_name": "TRULY_NEURAL_TIMER",
            }, "models/final/truly_neural_timer_best.pt")
            print(f"  Saved (fire_acc={100*fire_acc:.1f}%)")

        if fire_acc >= 0.95 and irq_acc >= 0.95:
            print("95%+ ACCURACY!")
            break

    print(f"\nBest fire accuracy: {100*best_acc:.1f}%")


if __name__ == "__main__":
    train()
