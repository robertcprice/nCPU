#!/usr/bin/env python3
"""
NEURAL GIC (Generic Interrupt Controller)
==========================================
Routes interrupts to appropriate handlers.

Instead of hardcoded interrupt routing, this neural network:
1. Takes interrupt number and system state
2. Decides priority and routing
3. Predicts which handler should run

This is neural interrupt handling!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


# Interrupt categories
IRQ_TYPES = {
    'TIMER': list(range(0, 16)),      # Timer interrupts (SGI)
    'IPI': list(range(16, 32)),       # Inter-processor interrupts
    'UART': [33, 34],                 # Serial console
    'DISK': list(range(48, 56)),      # Block devices
    'NET': list(range(56, 64)),       # Network
    'GPU': list(range(64, 72)),       # Graphics
    'USB': list(range(72, 80)),       # USB
    'MISC': list(range(80, 96)),      # Miscellaneous
}

IRQ_TO_TYPE = {}
for irq_type, irqs in IRQ_TYPES.items():
    for irq in irqs:
        IRQ_TO_TYPE[irq] = irq_type

TYPE_TO_IDX = {name: idx for idx, name in enumerate(IRQ_TYPES.keys())}
NUM_TYPES = len(IRQ_TYPES)

# Priority levels
PRIORITY_LEVELS = {
    'TIMER': 0,   # Highest
    'IPI': 1,
    'UART': 3,
    'DISK': 4,
    'NET': 4,
    'GPU': 5,
    'USB': 6,
    'MISC': 7,    # Lowest
}


class NeuralGIC(nn.Module):
    """
    Neural Generic Interrupt Controller.

    Routes interrupts based on:
    - Interrupt number
    - Current CPU state (running/idle)
    - Pending interrupt queue
    - Priority levels
    """

    def __init__(self, max_irq=256, max_pending=16, d_model=64):
        super().__init__()
        self.max_irq = max_irq
        self.num_types = NUM_TYPES

        # IRQ embedding
        self.irq_embed = nn.Embedding(max_irq, d_model)

        # Pending queue encoder (attention over pending IRQs)
        self.pending_encoder = nn.Sequential(
            nn.Linear(max_pending * d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

        # CPU state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(8, d_model),  # 8 state features
            nn.GELU(),
        )

        # Decision network
        self.decision = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )

        # Output heads
        self.route_head = nn.Linear(d_model, NUM_TYPES)  # Which handler type
        self.priority_head = nn.Linear(d_model, 8)       # Priority level
        self.mask_head = nn.Linear(d_model, 1)           # Should mask/ignore

    def forward(self, irq_num, pending_irqs, cpu_state):
        """
        Args:
            irq_num: [batch] - incoming IRQ number
            pending_irqs: [batch, max_pending] - pending IRQ queue
            cpu_state: [batch, 8] - CPU state features

        Returns:
            route_logits: [batch, num_types] - handler type
            priority: [batch, 8] - priority level
            mask: [batch] - should mask
        """
        batch = irq_num.shape[0]

        # Embed incoming IRQ
        irq_emb = self.irq_embed(irq_num.clamp(0, self.max_irq - 1))

        # Encode pending queue
        pending_clamped = pending_irqs.clamp(0, self.max_irq - 1)
        pending_emb = self.irq_embed(pending_clamped)  # [batch, max_pending, d_model]
        pending_flat = pending_emb.view(batch, -1)
        pending_enc = self.pending_encoder(pending_flat)

        # Encode CPU state
        state_enc = self.state_encoder(cpu_state)

        # Combine and decide
        combined = torch.cat([irq_emb, pending_enc, state_enc], dim=-1)
        h = self.decision(combined)

        # Outputs
        route_logits = self.route_head(h)
        priority = self.priority_head(h)
        mask = torch.sigmoid(self.mask_head(h).squeeze(-1))

        return route_logits, priority, mask


def generate_batch(batch_size, device):
    """Generate training batch."""
    irqs = []
    pending = []
    states = []
    targets = []
    priorities = []

    for _ in range(batch_size):
        # Random IRQ
        irq_type = random.choice(list(IRQ_TYPES.keys()))
        irq = random.choice(IRQ_TYPES[irq_type])

        # Random pending queue (mostly empty)
        pend = []
        for _ in range(16):
            if random.random() < 0.2:
                pend.append(random.randint(0, 95))
            else:
                pend.append(0)

        # Random CPU state
        state = [
            random.random(),  # CPU load
            float(random.random() < 0.3),  # idle
            float(random.random() < 0.8),  # interrupts enabled
            float(len([p for p in pend if p > 0])) / 16,  # pending ratio
            random.random(),  # timer counter normalized
            float(random.random() < 0.5),  # user mode
            float(random.random() < 0.9),  # not in critical section
            random.random(),  # random noise
        ]

        irqs.append(irq)
        pending.append(pend)
        states.append(state)
        targets.append(TYPE_TO_IDX[irq_type])
        priorities.append(PRIORITY_LEVELS[irq_type])

    return (
        torch.tensor(irqs, device=device),
        torch.tensor(pending, device=device),
        torch.tensor(states, dtype=torch.float32, device=device),
        torch.tensor(targets, device=device),
        torch.tensor(priorities, device=device)
    )


def train():
    print("=" * 60)
    print("NEURAL GIC (Interrupt Controller) TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"IRQ Types: {list(IRQ_TYPES.keys())}")

    # Create model
    model = NeuralGIC(max_irq=256, max_pending=16, d_model=64).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 512

    for epoch in range(100):
        model.train()
        total_loss = 0
        t0 = time.time()

        for _ in range(100):
            irqs, pending, states, targets, priorities = generate_batch(batch_size, device)

            optimizer.zero_grad()
            route_logits, priority_logits, mask = model(irqs, pending, states)

            loss_route = F.cross_entropy(route_logits, targets)
            loss_priority = F.cross_entropy(priority_logits, priorities)
            loss = loss_route + 0.5 * loss_priority

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        correct_route = 0
        correct_priority = 0
        total = 0

        with torch.no_grad():
            for _ in range(20):
                irqs, pending, states, targets, priorities = generate_batch(256, device)
                route_logits, priority_logits, _ = model(irqs, pending, states)

                route_preds = route_logits.argmax(dim=-1)
                priority_preds = priority_logits.argmax(dim=-1)

                correct_route += (route_preds == targets).sum().item()
                correct_priority += (priority_preds == priorities).sum().item()
                total += len(targets)

        route_acc = correct_route / total
        priority_acc = correct_priority / total
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} route_acc={100*route_acc:.1f}% priority_acc={100*priority_acc:.1f}% [{elapsed:.1f}s]")

        if route_acc > best_acc:
            best_acc = route_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": route_acc,
                "priority_accuracy": priority_acc,
                "op_name": "GIC",
                "architecture": "NeuralGIC",
                "irq_types": list(IRQ_TYPES.keys())
            }, "models/final/neural_gic_best.pt")
            print(f"  Saved (route_acc={100*route_acc:.1f}%)")

        if route_acc >= 0.99:
            print("99%+ ACCURACY!")
            break

    print(f"\nBest route accuracy: {100*best_acc:.1f}%")

    # Final verification
    print("\nFinal verification:")
    model.eval()

    test_irqs = [
        (5, 'TIMER'),
        (20, 'IPI'),
        (33, 'UART'),
        (50, 'DISK'),
        (60, 'NET'),
        (65, 'GPU'),
    ]

    type_names = list(IRQ_TYPES.keys())

    with torch.no_grad():
        for irq, expected_type in test_irqs:
            irq_t = torch.tensor([irq], device=device)
            pending_t = torch.zeros(1, 16, dtype=torch.long, device=device)
            state_t = torch.randn(1, 8, device=device)

            route_logits, priority_logits, mask = model(irq_t, pending_t, state_t)
            pred_idx = route_logits.argmax(dim=-1).item()
            pred_type = type_names[pred_idx]
            pred_priority = priority_logits.argmax(dim=-1).item()

            status = "OK" if pred_type == expected_type else f"GOT {pred_type}"
            print(f"  IRQ {irq} → {expected_type} (priority {PRIORITY_LEVELS[expected_type]}): {status} (pred priority {pred_priority})")


if __name__ == "__main__":
    train()
