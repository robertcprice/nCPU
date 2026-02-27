#!/usr/bin/env python3
"""
TRULY NEURAL ORCHESTRATOR - LEARNED BEHAVIOR
==============================================
The network LEARNS the CPU state machine from examples.
NO pre-defined phase→action mappings. NO lookup tables.

The network learns:
- State transitions (which phase comes next)
- Action selection based on current state
- The fetch-decode-execute cycle pattern

This is a LEARNED state machine!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


# State and action definitions
PHASES = ['FETCH', 'DECODE', 'EXECUTE', 'WRITEBACK', 'INTERRUPT']
ACTIONS = ['FETCH_INSTR', 'DECODE_INSTR', 'EXECUTE_ALU', 'EXECUTE_MEM',
           'EXECUTE_BRANCH', 'WRITE_REG', 'HANDLE_IRQ', 'IDLE']

NUM_PHASES = len(PHASES)
NUM_ACTIONS = len(ACTIONS)


class TrulyNeuralOrchestratorLearned(nn.Module):
    """
    Orchestrator that LEARNS state machine behavior.

    The network learns:
    - phase + context → action (what to do)
    - phase + context → next_phase (state transition)
    """

    def __init__(self, num_phases=NUM_PHASES, num_actions=NUM_ACTIONS,
                 context_dim=16, hidden_dim=128):
        super().__init__()
        self.num_phases = num_phases
        self.num_actions = num_actions

        # Current phase state IN WEIGHTS
        self.phase_state = nn.Parameter(torch.zeros(num_phases))

        # LEARNED action selector
        self.action_net = nn.Sequential(
            nn.Linear(num_phases + context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # LEARNED state transition (next phase predictor)
        self.transition_net = nn.Sequential(
            nn.Linear(num_phases + num_actions + context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_phases),
        )

        # LEARNED context encoder (for IRQ, stalls, etc)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, context_dim),
        )

    def select_action(self, phase_onehot, context):
        """
        LEARN which action to take given phase and context.
        """
        encoded_ctx = self.context_encoder(context)
        action_input = torch.cat([phase_onehot, encoded_ctx], dim=-1)
        action_logits = self.action_net(action_input)
        return action_logits

    def predict_next_phase(self, phase_onehot, action_onehot, context):
        """
        LEARN state transition - what's the next phase?
        """
        encoded_ctx = self.context_encoder(context)
        trans_input = torch.cat([phase_onehot, action_onehot, encoded_ctx], dim=-1)
        next_phase_logits = self.transition_net(trans_input)
        return next_phase_logits

    def step(self, context):
        """
        Execute one orchestration step - select action and transition.
        """
        batch = context.shape[0]

        # Current phase
        phase_probs = F.softmax(self.phase_state, dim=-1).unsqueeze(0).expand(batch, -1)

        # Select action
        action_logits = self.select_action(phase_probs, context)
        action_probs = F.softmax(action_logits, dim=-1)

        # Predict next phase
        next_phase_logits = self.predict_next_phase(phase_probs, action_probs, context)
        next_phase_probs = F.softmax(next_phase_logits, dim=-1)

        # Update phase state (learned transition)
        next_phase_mean = next_phase_probs.mean(dim=0)
        phase_logits = torch.log(next_phase_mean.clamp(0.01, 0.99) / (1 - next_phase_mean.clamp(0.01, 0.99)))
        self.phase_state.data = 0.7 * self.phase_state.data + 0.3 * phase_logits

        return action_logits, next_phase_logits


def generate_batch(batch_size, device):
    """
    Generate training examples for the CPU state machine.

    The fetch-decode-execute cycle:
    FETCH → DECODE → EXECUTE → WRITEBACK → (FETCH or INTERRUPT)
    """
    phases = []
    contexts = []
    expected_actions = []
    expected_next_phases = []

    for _ in range(batch_size):
        phase = random.randint(0, NUM_PHASES - 1)
        phase_onehot = F.one_hot(torch.tensor(phase), num_classes=NUM_PHASES).float()

        # Context: [has_irq, is_branch, is_mem, is_alu, stall, ...]
        context = torch.zeros(16)
        has_irq = random.random() < 0.1
        is_branch = random.random() < 0.2
        is_mem = random.random() < 0.3
        context[0] = float(has_irq)
        context[1] = float(is_branch)
        context[2] = float(is_mem)

        # Ground truth based on CPU state machine
        if phase == 0:  # FETCH
            action = 0  # FETCH_INSTR
            next_phase = 1  # → DECODE
        elif phase == 1:  # DECODE
            action = 1  # DECODE_INSTR
            next_phase = 2  # → EXECUTE
        elif phase == 2:  # EXECUTE
            if is_branch:
                action = 4  # EXECUTE_BRANCH
            elif is_mem:
                action = 3  # EXECUTE_MEM
            else:
                action = 2  # EXECUTE_ALU
            next_phase = 3  # → WRITEBACK
        elif phase == 3:  # WRITEBACK
            action = 5  # WRITE_REG
            if has_irq:
                next_phase = 4  # → INTERRUPT
            else:
                next_phase = 0  # → FETCH (loop)
        else:  # INTERRUPT
            action = 6  # HANDLE_IRQ
            next_phase = 0  # → FETCH

        phases.append(phase_onehot)
        contexts.append(context)
        expected_actions.append(action)
        expected_next_phases.append(next_phase)

    return (
        torch.stack(phases).to(device),
        torch.stack(contexts).to(device),
        torch.tensor(expected_actions, device=device),
        torch.tensor(expected_next_phases, device=device),
    )


def train():
    print("=" * 60)
    print("TRULY NEURAL ORCHESTRATOR - LEARNED BEHAVIOR")
    print("=" * 60)
    print(f"Device: {device}")
    print("Network LEARNS CPU state machine from examples!")
    print("NO pre-defined mappings. Pure learning.")
    print(f"Phases: {PHASES}")
    print(f"Actions: {ACTIONS}")

    model = TrulyNeuralOrchestratorLearned(
        num_phases=NUM_PHASES,
        num_actions=NUM_ACTIONS,
        context_dim=16,
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
            phases, contexts, exp_actions, exp_next_phases = generate_batch(batch_size, device)

            optimizer.zero_grad()

            # Predict action
            action_logits = model.select_action(phases, contexts)
            action_probs = F.softmax(action_logits, dim=-1)

            # Predict next phase
            next_phase_logits = model.predict_next_phase(phases, action_probs, contexts)

            # Losses
            loss_action = F.cross_entropy(action_logits, exp_actions)
            loss_phase = F.cross_entropy(next_phase_logits, exp_next_phases)

            loss = loss_action + loss_phase
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        action_correct = 0
        phase_correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(20):
                phases, contexts, exp_actions, exp_next_phases = generate_batch(256, device)

                action_logits = model.select_action(phases, contexts)
                action_preds = action_logits.argmax(dim=-1)

                action_probs = F.softmax(action_logits, dim=-1)
                next_phase_logits = model.predict_next_phase(phases, action_probs, contexts)
                phase_preds = next_phase_logits.argmax(dim=-1)

                action_correct += (action_preds == exp_actions).sum().item()
                phase_correct += (phase_preds == exp_next_phases).sum().item()
                total += len(exp_actions)

        action_acc = action_correct / total
        phase_acc = phase_correct / total
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} action_acc={100*action_acc:.1f}% phase_acc={100*phase_acc:.1f}% [{elapsed:.1f}s]")

        if action_acc > best_acc:
            best_acc = action_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "action_accuracy": action_acc,
                "phase_accuracy": phase_acc,
                "op_name": "TRULY_NEURAL_ORCHESTRATOR_LEARNED",
                "phases": PHASES,
                "actions": ACTIONS,
            }, "models/final/truly_neural_orchestrator_learned_best.pt")
            print(f"  Saved (action_acc={100*action_acc:.1f}%)")

        if action_acc >= 0.99 and phase_acc >= 0.99:
            print("99%+ ACCURACY ON BOTH!")
            break

    print(f"\nBest action accuracy: {100*best_acc:.1f}%")

    # Final verification
    print("\nFinal verification - learned state machine:")
    model.eval()

    with torch.no_grad():
        print("  Simulating fetch-decode-execute cycle:")
        # Start at FETCH
        for i in range(10):
            phase_idx = i % NUM_PHASES
            phase_onehot = F.one_hot(torch.tensor([phase_idx]), num_classes=NUM_PHASES).float().to(device)
            context = torch.zeros(1, 16, device=device)

            action_logits = model.select_action(phase_onehot, context)
            action = action_logits.argmax(dim=-1).item()

            action_probs = F.softmax(action_logits, dim=-1)
            next_phase_logits = model.predict_next_phase(phase_onehot, action_probs, context)
            next_phase = next_phase_logits.argmax(dim=-1).item()

            print(f"    {PHASES[phase_idx]} → {ACTIONS[action]} → {PHASES[next_phase]}")


if __name__ == "__main__":
    train()
