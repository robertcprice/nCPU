#!/usr/bin/env python3
"""
TRULY NEURAL GIC (Generic Interrupt Controller)
================================================
Interrupt routing stored IN THE NETWORK WEIGHTS.

Like the Truly Neural Register File:
- IRQ→Handler mappings stored as nn.Parameter
- Priority levels stored as nn.Parameter
- Pending queue state stored as nn.Parameter
- ALL state is neural - no Python data structures!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


# IRQ categories (for training data generation only)
IRQ_HANDLERS = ['TIMER', 'IPI', 'UART', 'DISK', 'NET', 'GPU', 'USB', 'MISC']
NUM_HANDLERS = len(IRQ_HANDLERS)

IRQ_RANGES = {
    'TIMER': list(range(0, 16)),
    'IPI': list(range(16, 32)),
    'UART': [33, 34],
    'DISK': list(range(48, 56)),
    'NET': list(range(56, 64)),
    'GPU': list(range(64, 72)),
    'USB': list(range(72, 80)),
    'MISC': list(range(80, 96)),
}

IRQ_TO_HANDLER = {}
for handler, irqs in IRQ_RANGES.items():
    for irq in irqs:
        IRQ_TO_HANDLER[irq] = IRQ_HANDLERS.index(handler)

PRIORITY_MAP = {
    'TIMER': 0, 'IPI': 1, 'UART': 3, 'DISK': 4,
    'NET': 4, 'GPU': 5, 'USB': 6, 'MISC': 7,
}


class TrulyNeuralGIC(nn.Module):
    """
    GIC where ALL state is stored IN the neural network weights.

    Architecture:
    - irq_routing: nn.Parameter [max_irqs, num_handlers] - routing IN weights
    - irq_priorities: nn.Parameter [max_irqs] - priority levels IN weights
    - pending_state: nn.Parameter [max_irqs] - pending queue IN weights
    - mask_state: nn.Parameter [max_irqs] - IRQ masks IN weights

    Route: irq → attention → handler + priority
    State updates: Hebbian updates to pending/mask
    """

    def __init__(self, max_irqs=256, num_handlers=NUM_HANDLERS, key_dim=64):
        super().__init__()
        self.max_irqs = max_irqs
        self.num_handlers = num_handlers
        self.key_dim = key_dim

        # === ALL STATE IN WEIGHTS ===
        # IRQ → handler routing table
        self.irq_routing = nn.Parameter(torch.zeros(max_irqs, num_handlers))

        # Priority per IRQ (0 = highest, 7 = lowest)
        self.irq_priorities = nn.Parameter(torch.zeros(max_irqs, 8))

        # Pending state (which IRQs are pending)
        self.pending_state = nn.Parameter(torch.zeros(max_irqs))

        # Mask state (which IRQs are masked)
        self.mask_state = nn.Parameter(torch.zeros(max_irqs))

        # Keys for attention
        self.irq_keys = nn.Parameter(torch.randn(max_irqs, key_dim) * 0.1)

        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(max_irqs + 8, key_dim),  # IRQ one-hot + CPU state
            nn.GELU(),
            nn.Linear(key_dim, key_dim),
        )

        # Temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Update learning rate
        self.update_lr = nn.Parameter(torch.tensor(0.1))

    def _irq_to_onehot(self, irq_num):
        """Convert IRQ number to one-hot."""
        batch = irq_num.shape[0]
        onehot = torch.zeros(batch, self.max_irqs, device=irq_num.device)
        valid = irq_num < self.max_irqs
        onehot[valid, irq_num[valid].long()] = 1.0
        return onehot

    def route(self, irq_num, cpu_state=None):
        """
        Route IRQ to handler.

        Args:
            irq_num: [batch] - IRQ numbers
            cpu_state: [batch, 8] - optional CPU state

        Returns:
            handler_logits: [batch, num_handlers]
            priority_logits: [batch, 8]
            should_handle: [batch] - not masked and valid
        """
        batch = irq_num.shape[0]

        # Create query
        irq_onehot = self._irq_to_onehot(irq_num)

        if cpu_state is None:
            cpu_state = torch.zeros(batch, 8, device=irq_num.device)

        query_input = torch.cat([irq_onehot, cpu_state], dim=-1)
        query = self.query_encoder(query_input)

        # Attention over IRQ keys
        similarity = torch.matmul(query, self.irq_keys.T)
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(similarity / temp, dim=-1)

        # Read routing
        handler_logits = torch.matmul(attention, self.irq_routing)

        # Read priority
        priority_logits = torch.matmul(attention, self.irq_priorities)

        # Check mask (should handle if not masked)
        mask_value = torch.matmul(attention, self.mask_state.unsqueeze(-1)).squeeze(-1)
        should_handle = torch.sigmoid(-mask_value)  # Inverted: masked = don't handle

        return handler_logits, priority_logits, should_handle

    def raise_irq(self, irq_num):
        """Mark IRQ as pending (Hebbian update)."""
        irq_onehot = self._irq_to_onehot(irq_num)
        update = irq_onehot.sum(dim=0)  # Sum across batch
        lr = torch.clamp(self.update_lr.abs(), 0.01, 1.0)
        self.pending_state.data = torch.clamp(
            self.pending_state.data + lr * update, 0, 1
        )

    def ack_irq(self, irq_num):
        """Clear IRQ pending state."""
        irq_onehot = self._irq_to_onehot(irq_num)
        update = irq_onehot.sum(dim=0)
        lr = torch.clamp(self.update_lr.abs(), 0.01, 1.0)
        self.pending_state.data = torch.clamp(
            self.pending_state.data - lr * update, 0, 1
        )

    def learn_routing(self, irq_num, target_handler, target_priority):
        """Learn IRQ → handler + priority mapping."""
        batch = irq_num.shape[0]

        # Handler target
        handler_target = torch.zeros(batch, self.num_handlers, device=irq_num.device)
        handler_target[torch.arange(batch), target_handler] = 1.0

        # Priority target
        priority_target = torch.zeros(batch, 8, device=irq_num.device)
        priority_target[torch.arange(batch), target_priority] = 1.0

        # IRQ attention (direct)
        irq_attention = self._irq_to_onehot(irq_num)

        # Hebbian updates
        lr = torch.clamp(self.update_lr.abs(), 0.01, 1.0)

        routing_update = torch.matmul(irq_attention.T, handler_target)
        self.irq_routing.data = self.irq_routing.data + lr * routing_update

        priority_update = torch.matmul(irq_attention.T, priority_target)
        self.irq_priorities.data = self.irq_priorities.data + lr * priority_update

    def get_pending(self):
        """Get highest priority pending IRQ."""
        # Combine pending state with priority
        pending = torch.sigmoid(self.pending_state)
        priority = self.irq_priorities.argmax(dim=-1).float()

        # Score = pending * (8 - priority)  (higher priority = lower number = higher score)
        score = pending * (8 - priority / 7)

        best_irq = score.argmax()
        best_score = score[best_irq]

        return best_irq, best_score


def generate_batch(batch_size, device):
    """Generate training batch."""
    irqs = []
    handlers = []
    priorities = []
    cpu_states = []

    for _ in range(batch_size):
        handler_name = random.choice(list(IRQ_RANGES.keys()))
        irq = random.choice(IRQ_RANGES[handler_name])
        handler = IRQ_HANDLERS.index(handler_name)
        priority = PRIORITY_MAP[handler_name]

        irqs.append(irq)
        handlers.append(handler)
        priorities.append(priority)
        cpu_states.append([random.random() for _ in range(8)])

    return (
        torch.tensor(irqs, device=device),
        torch.tensor(handlers, device=device),
        torch.tensor(priorities, device=device),
        torch.tensor(cpu_states, dtype=torch.float32, device=device)
    )


def train():
    print("=" * 60)
    print("TRULY NEURAL GIC TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("ALL interrupt state stored IN NETWORK WEIGHTS!")
    print(f"Handlers: {IRQ_HANDLERS}")

    model = TrulyNeuralGIC(
        max_irqs=256,
        num_handlers=NUM_HANDLERS,
        key_dim=64
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 512

    for epoch in range(100):
        model.train()
        total_loss = 0
        t0 = time.time()

        for step in range(100):
            irqs, handlers, priorities, cpu_states = generate_batch(batch_size, device)

            # Learning phase
            if step < 10 or random.random() < 0.1:
                model.learn_routing(irqs, handlers, priorities)

            optimizer.zero_grad()

            # Route
            handler_logits, priority_logits, should_handle = model.route(irqs, cpu_states)

            # Losses
            loss_handler = F.cross_entropy(handler_logits, handlers)
            loss_priority = F.cross_entropy(priority_logits, priorities)
            loss = loss_handler + 0.5 * loss_priority

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        correct_handler = 0
        correct_priority = 0
        total = 0

        with torch.no_grad():
            for _ in range(20):
                irqs, handlers, priorities, cpu_states = generate_batch(256, device)
                handler_logits, priority_logits, _ = model.route(irqs, cpu_states)

                h_preds = handler_logits.argmax(dim=-1)
                p_preds = priority_logits.argmax(dim=-1)

                correct_handler += (h_preds == handlers).sum().item()
                correct_priority += (p_preds == priorities).sum().item()
                total += len(handlers)

        h_acc = correct_handler / total
        p_acc = correct_priority / total
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} handler_acc={100*h_acc:.1f}% priority_acc={100*p_acc:.1f}% [{elapsed:.1f}s]")

        if h_acc > best_acc:
            best_acc = h_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "handler_accuracy": h_acc,
                "priority_accuracy": p_acc,
                "op_name": "TRULY_NEURAL_GIC",
                "handlers": IRQ_HANDLERS
            }, "models/final/truly_neural_gic_best.pt")
            print(f"  Saved (handler_acc={100*h_acc:.1f}%)")

        if h_acc >= 0.99:
            print("99%+ ACCURACY!")
            break

    print(f"\nBest handler accuracy: {100*best_acc:.1f}%")

    # Final verification
    print("\nFinal verification:")
    model.eval()
    test_irqs = [
        (5, 'TIMER', 0), (20, 'IPI', 1), (33, 'UART', 3),
        (50, 'DISK', 4), (60, 'NET', 4), (65, 'GPU', 5),
    ]

    with torch.no_grad():
        for irq, expected_handler, expected_priority in test_irqs:
            irq_t = torch.tensor([irq], device=device)
            handler_logits, priority_logits, _ = model.route(irq_t)

            pred_h = IRQ_HANDLERS[handler_logits.argmax(dim=-1).item()]
            pred_p = priority_logits.argmax(dim=-1).item()

            h_status = "OK" if pred_h == expected_handler else f"GOT {pred_h}"
            p_status = "OK" if pred_p == expected_priority else f"GOT {pred_p}"
            print(f"  IRQ {irq} → {expected_handler}(p={expected_priority}): {h_status}, priority {p_status}")


if __name__ == "__main__":
    train()
