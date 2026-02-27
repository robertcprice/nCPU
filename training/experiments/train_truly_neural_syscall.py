#!/usr/bin/env python3
"""
TRULY NEURAL SYSCALL ROUTER
============================
Syscall routing tables stored IN THE NETWORK WEIGHTS.

Like the Truly Neural Register File:
- Syscall→Subsystem mappings stored as nn.Parameter
- Routing = attention-based lookup
- Learning = Hebbian updates to routing weights
- NO hardcoded switch statements - everything is neural!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


# Subsystem definitions (for training data generation only)
SUBSYSTEMS = ['PROC', 'MEM', 'IO', 'FS', 'NET', 'SEC', 'TIME', 'MISC']
NUM_SUBSYSTEMS = len(SUBSYSTEMS)

# Training data - syscall → subsystem (used to generate training examples)
SYSCALL_CATEGORIES = {
    'PROC': [93, 94, 172, 174, 175, 176, 177, 178, 220, 221, 260, 435],
    'MEM': [214, 215, 222, 226, 227, 228, 229, 230, 231, 232, 233, 234],
    'IO': [63, 64, 65, 66, 67, 68, 62, 17, 23, 24, 25, 29],
    'FS': [56, 57, 48, 49, 79, 80, 78, 34, 35, 83, 84, 85, 86, 87, 88],
    'NET': [198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209],
    'SEC': [146, 147, 148, 149, 150, 151, 152, 153, 160, 161],
    'TIME': [101, 113, 114, 115, 169, 170],
    'MISC': [278, 279, 280, 281, 282, 283, 284],
}

SYSCALL_TO_SUBSYSTEM = {}
for sub, syscalls in SYSCALL_CATEGORIES.items():
    for sc in syscalls:
        SYSCALL_TO_SUBSYSTEM[sc] = SUBSYSTEMS.index(sub)


class TrulyNeuralSyscallRouter(nn.Module):
    """
    Syscall router where routing table is stored IN the neural network weights.

    Architecture:
    - routing_table: nn.Parameter [max_syscalls, num_subsystems] - routing stored IN weights
    - syscall_keys: nn.Parameter [max_syscalls, key_dim] - learned keys
    - subsystem_values: nn.Parameter [num_subsystems, value_dim] - learned subsystem embeddings

    Route: syscall_num → query → attention → subsystem selection
    Learn: Hebbian update of routing_table
    """

    def __init__(self, max_syscalls=512, num_subsystems=NUM_SUBSYSTEMS, key_dim=64):
        super().__init__()
        self.max_syscalls = max_syscalls
        self.num_subsystems = num_subsystems
        self.key_dim = key_dim

        # === ROUTING TABLE STORED IN WEIGHTS ===
        # routing_table[i, j] = strength of syscall i → subsystem j
        self.routing_table = nn.Parameter(
            torch.zeros(max_syscalls, num_subsystems)
        )

        # Keys for attention-based syscall lookup
        self.syscall_keys = nn.Parameter(
            torch.randn(max_syscalls, key_dim) * 0.1
        )

        # Subsystem embeddings (what each subsystem "looks like")
        self.subsystem_embeddings = nn.Parameter(
            torch.randn(num_subsystems, key_dim) * 0.1
        )

        # Query encoder: syscall one-hot + args → key space
        self.query_encoder = nn.Sequential(
            nn.Linear(max_syscalls + 6 * 64, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
        )

        # Temperature for attention
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Write learning rate
        self.write_lr = nn.Parameter(torch.tensor(0.1))

    def _syscall_to_onehot(self, syscall_num):
        """Convert syscall number to one-hot."""
        batch = syscall_num.shape[0]
        onehot = torch.zeros(batch, self.max_syscalls, device=syscall_num.device)
        valid = syscall_num < self.max_syscalls
        onehot[valid, syscall_num[valid].long()] = 1.0
        return onehot

    def route(self, syscall_num, arg_bits=None):
        """
        Route syscall to subsystem.

        Args:
            syscall_num: [batch] - syscall numbers
            arg_bits: [batch, 6, 64] - optional argument bits

        Returns:
            subsystem_logits: [batch, num_subsystems]
            routing_confidence: [batch]
        """
        batch = syscall_num.shape[0]

        # Create query from syscall + args
        syscall_onehot = self._syscall_to_onehot(syscall_num)

        if arg_bits is not None:
            arg_flat = arg_bits.view(batch, -1)
            query_input = torch.cat([syscall_onehot, arg_flat], dim=-1)
        else:
            query_input = torch.cat([
                syscall_onehot,
                torch.zeros(batch, 6 * 64, device=syscall_num.device)
            ], dim=-1)

        # Encode as query
        query = self.query_encoder(query_input)  # [B, key_dim]

        # Attention over syscall keys
        similarity = torch.matmul(query, self.syscall_keys.T)  # [B, max_syscalls]
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(similarity / temp, dim=-1)  # [B, max_syscalls]

        # Read routing from table via attention
        # routing_table: [max_syscalls, num_subsystems]
        subsystem_logits = torch.matmul(attention, self.routing_table)  # [B, num_subsystems]

        # Confidence is max attention weight
        confidence = attention.max(dim=-1).values

        return subsystem_logits, confidence

    def learn_route(self, syscall_num, target_subsystem):
        """
        Learn a syscall → subsystem mapping (Hebbian update).

        Args:
            syscall_num: [batch] - syscall numbers
            target_subsystem: [batch] - target subsystem indices
        """
        batch = syscall_num.shape[0]

        # Create target one-hot
        target_onehot = torch.zeros(batch, self.num_subsystems, device=syscall_num.device)
        target_onehot[torch.arange(batch), target_subsystem] = 1.0

        # Create syscall attention (direct for learning)
        syscall_attention = self._syscall_to_onehot(syscall_num)

        # Hebbian update: table += lr * syscall.T @ target
        update = torch.matmul(syscall_attention.T, target_onehot)
        lr = torch.clamp(self.write_lr.abs(), 0.01, 1.0)
        self.routing_table.data = self.routing_table.data + lr * update


def generate_batch(batch_size, device):
    """Generate training batch."""
    syscalls = []
    targets = []
    args = []

    for _ in range(batch_size):
        # Pick random known syscall
        sub_name = random.choice(list(SYSCALL_CATEGORIES.keys()))
        syscall = random.choice(SYSCALL_CATEGORIES[sub_name])
        target = SUBSYSTEMS.index(sub_name)

        syscalls.append(syscall)
        targets.append(target)

        # Random args
        arg_list = [
            torch.tensor([(random.randint(0, 1000) >> i) & 1 for i in range(64)], dtype=torch.float32)
            for _ in range(6)
        ]
        args.append(torch.stack(arg_list))

    return (
        torch.tensor(syscalls, device=device),
        torch.tensor(targets, device=device),
        torch.stack(args).to(device)
    )


def train():
    print("=" * 60)
    print("TRULY NEURAL SYSCALL ROUTER TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("Routing table stored IN NETWORK WEIGHTS!")
    print(f"Subsystems: {SUBSYSTEMS}")

    model = TrulyNeuralSyscallRouter(
        max_syscalls=512,
        num_subsystems=NUM_SUBSYSTEMS,
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
            syscalls, targets, args = generate_batch(batch_size, device)

            # Learning phase - teach the routing
            if step < 10 or random.random() < 0.1:
                model.learn_route(syscalls, targets)

            optimizer.zero_grad()

            # Route
            logits, confidence = model.route(syscalls, args)

            # Loss
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(20):
                syscalls, targets, args = generate_batch(256, device)
                logits, _ = model.route(syscalls, args)
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += len(targets)

        acc = correct / total
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} acc={100*acc:.1f}% [{elapsed:.1f}s]")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": acc,
                "op_name": "TRULY_NEURAL_SYSCALL_ROUTER",
                "subsystems": SUBSYSTEMS
            }, "models/final/truly_neural_syscall_router_best.pt")
            print(f"  Saved (acc={100*acc:.1f}%)")

        if acc >= 0.99:
            print("99%+ ACCURACY!")
            break

    print(f"\nBest accuracy: {100*best_acc:.1f}%")

    # Final verification
    print("\nFinal verification:")
    model.eval()
    test_syscalls = [
        (63, 'IO'), (64, 'IO'), (93, 'PROC'), (214, 'MEM'),
        (222, 'MEM'), (56, 'FS'), (57, 'FS'), (278, 'MISC'),
    ]

    with torch.no_grad():
        for sc, expected_sub in test_syscalls:
            sc_t = torch.tensor([sc], device=device)
            logits, conf = model.route(sc_t)
            pred_idx = logits.argmax(dim=-1).item()
            pred_sub = SUBSYSTEMS[pred_idx]
            status = "OK" if pred_sub == expected_sub else f"GOT {pred_sub}"
            print(f"  syscall {sc} → {expected_sub}: {status} (conf={conf.item():.2f})")


if __name__ == "__main__":
    train()
