#!/usr/bin/env python3
"""
TRULY NEURAL SYSTEM ORCHESTRATOR V2
====================================
Uses the same pre-population + attention pattern as other successful models.

Key insight: The orchestrator just needs to learn phase→action mappings.
We pre-populate these mappings and train to recall them.

Phases: FETCH, DECODE, EXECUTE, SYSCALL, IRQ, IDLE
Actions: fetch, decode, execute_alu, execute_mem, execute_branch,
         syscall, irq_check, irq_handle, context_switch, idle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


# Phase and action definitions
PHASES = ['FETCH', 'DECODE', 'EXECUTE', 'SYSCALL', 'IRQ', 'IDLE']
ACTIONS = ['FETCH', 'DECODE', 'EXECUTE_ALU', 'EXECUTE_MEM', 'EXECUTE_BRANCH',
           'SYSCALL', 'IRQ_CHECK', 'IRQ_HANDLE', 'CONTEXT_SWITCH', 'IDLE']

NUM_PHASES = len(PHASES)
NUM_ACTIONS = len(ACTIONS)


class TrulyNeuralOrchestratorV2(nn.Module):
    """
    Orchestrator V2 - fixed phase→action mappings IN weights.

    Like the syscall router and GIC:
    - Phase→action routing table stored as nn.Parameter
    - Lookup via attention
    """

    def __init__(self, num_phases=NUM_PHASES, num_actions=NUM_ACTIONS, key_dim=32):
        super().__init__()
        self.num_phases = num_phases
        self.num_actions = num_actions
        self.key_dim = key_dim

        # === ROUTING TABLE IN WEIGHTS ===
        # routing_table[phase, action] = strength of phase→action mapping
        self.routing_table = nn.Parameter(torch.zeros(num_phases, num_actions))

        # Phase keys for attention
        self.phase_keys = nn.Parameter(torch.randn(num_phases, key_dim) * 0.1)

        # Query encoder: phase + context → key space
        self.query_encoder = nn.Sequential(
            nn.Linear(num_phases + 16, key_dim),  # phase one-hot + context
            nn.GELU(),
            nn.Linear(key_dim, key_dim),
        )

        self.temperature = nn.Parameter(torch.tensor(0.5))

    def select_action(self, phase_onehot, context=None):
        """
        Select action based on phase.

        Args:
            phase_onehot: [batch, num_phases] - current phase
            context: [batch, 16] - optional context (irq pending, etc)

        Returns:
            action_logits: [batch, num_actions]
            confidence: [batch]
        """
        batch = phase_onehot.shape[0]

        if context is None:
            context = torch.zeros(batch, 16, device=phase_onehot.device)

        # Create query
        query_input = torch.cat([phase_onehot, context], dim=-1)
        query = self.query_encoder(query_input)

        # Attention over phases
        key_sim = torch.matmul(query, self.phase_keys.T)

        # Also use direct phase matching
        stored_phases = F.one_hot(torch.arange(self.num_phases, device=phase_onehot.device), num_classes=self.num_phases).float()
        phase_sim = torch.matmul(phase_onehot, stored_phases.T)

        combined_sim = key_sim + 2.0 * phase_sim

        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(combined_sim / temp, dim=-1)

        # Read action logits from routing table
        action_logits = torch.matmul(attention, self.routing_table)

        confidence = attention.max(dim=-1).values

        return action_logits, confidence

    def populate_routing(self, phase_action_map):
        """
        Populate routing table.

        Args:
            phase_action_map: dict mapping phase_idx → action_idx
        """
        with torch.no_grad():
            for phase_idx, action_idx in phase_action_map.items():
                # Set strong weight for this mapping
                self.routing_table.data[phase_idx, action_idx] = 3.0

            # Update keys to match phases
            for phase_idx in range(self.num_phases):
                phase_onehot = F.one_hot(torch.tensor(phase_idx), num_classes=self.num_phases).float().unsqueeze(0).to(device)
                context = torch.zeros(1, 16, device=device)
                query_input = torch.cat([phase_onehot, context], dim=-1)
                self.phase_keys.data[phase_idx] = self.query_encoder(query_input).detach().squeeze(0)


# Fixed phase→action mappings
PHASE_ACTION_MAP = {
    0: 0,  # FETCH phase → FETCH action
    1: 1,  # DECODE phase → DECODE action
    2: 2,  # EXECUTE phase → EXECUTE_ALU (default)
    3: 5,  # SYSCALL phase → SYSCALL action
    4: 6,  # IRQ phase → IRQ_CHECK action
    5: 9,  # IDLE phase → IDLE action
}


def generate_batch(batch_size, device):
    """Generate training batch."""
    phases = []
    contexts = []
    expected_actions = []

    for _ in range(batch_size):
        phase_idx = random.randint(0, NUM_PHASES - 1)
        phase_onehot = F.one_hot(torch.tensor(phase_idx), num_classes=NUM_PHASES).float()

        # Context: random for now
        context = torch.rand(16)

        # Expected action based on fixed mapping
        if phase_idx == 2:  # EXECUTE phase - can be different execute types
            action_idx = random.choice([2, 3, 4])  # ALU, MEM, or BRANCH
        elif phase_idx == 4:  # IRQ phase
            has_irq = context[0] > 0.5
            action_idx = 7 if has_irq else 6  # HANDLE if IRQ pending, else CHECK
        else:
            action_idx = PHASE_ACTION_MAP[phase_idx]

        phases.append(phase_onehot)
        contexts.append(context)
        expected_actions.append(action_idx)

    return (
        torch.stack(phases).to(device),
        torch.stack(contexts).to(device),
        torch.tensor(expected_actions, device=device),
    )


def train():
    print("=" * 60)
    print("TRULY NEURAL ORCHESTRATOR V2 TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("Fixed phase→action mappings IN NETWORK WEIGHTS!")
    print(f"Phases: {PHASES}")
    print(f"Actions: {ACTIONS}")

    model = TrulyNeuralOrchestratorV2(
        num_phases=NUM_PHASES,
        num_actions=NUM_ACTIONS,
        key_dim=32
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Pre-populate routing table
    print("Populating routing table in weights...")
    model.populate_routing(PHASE_ACTION_MAP)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 512

    for epoch in range(100):
        model.train()
        total_loss = 0
        t0 = time.time()

        for _ in range(100):
            phases, contexts, expected_actions = generate_batch(batch_size, device)

            optimizer.zero_grad()

            action_logits, confidence = model.select_action(phases, contexts)

            loss = F.cross_entropy(action_logits, expected_actions)
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
                phases, contexts, expected_actions = generate_batch(256, device)
                action_logits, _ = model.select_action(phases, contexts)
                pred_actions = action_logits.argmax(dim=-1)
                correct += (pred_actions == expected_actions).sum().item()
                total += len(expected_actions)

        acc = correct / total
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} action_acc={100*acc:.1f}% [{elapsed:.1f}s]")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "action_accuracy": acc,
                "op_name": "TRULY_NEURAL_ORCHESTRATOR_V2",
                "phases": PHASES,
                "actions": ACTIONS,
            }, "models/final/truly_neural_orchestrator_v2_best.pt")
            print(f"  Saved (action_acc={100*acc:.1f}%)")

        if acc >= 0.99:
            print("99%+ ACCURACY!")
            break

    print(f"\nBest action accuracy: {100*best_acc:.1f}%")

    # Final verification
    print("\nFinal verification - phase→action:")
    model.eval()
    with torch.no_grad():
        for phase_idx, phase_name in enumerate(PHASES):
            phase_onehot = F.one_hot(torch.tensor([phase_idx]), num_classes=NUM_PHASES).float().to(device)
            action_logits, conf = model.select_action(phase_onehot)
            pred_action = action_logits.argmax(dim=-1).item()
            print(f"  {phase_name} → {ACTIONS[pred_action]} (conf={conf.item():.2f})")


if __name__ == "__main__":
    train()
