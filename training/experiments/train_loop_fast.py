#!/usr/bin/env python3
"""
Fast Neural Loop Training - Practical Approach

Key insight: For loop detection, we need to identify:
1. Loop TYPE: Is it countdown, count-up, or mem-fill? (from instruction pattern)
2. Counter REGISTER: Which register has the "loop count"? (from register VALUES)
3. ITERATIONS: How many times will it run? (from counter value)

The register values tell us MOST of what we need!
- Countdown: counter reg has value in [10, 100000]
- Count-up: counter reg + limit reg, difference tells iterations
- Mem-fill: ptr reg + counter reg
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


class FastLoopDetector(nn.Module):
    """
    Lightweight loop detector focusing on what matters:
    - Instruction OPCODE patterns (not all 32 bits)
    - Register VALUE patterns (which regs have "loop-like" values)
    """

    def __init__(self, max_body_len: int = 8):
        super().__init__()
        self.max_body_len = max_body_len

        # ═══════════════════════════════════════════════════════════════
        # INSTRUCTION ENCODER - Focus on opcodes (bits 21-31)
        # ═══════════════════════════════════════════════════════════════
        self.opcode_embed = nn.Sequential(
            nn.Linear(11 * max_body_len, 64),  # 11 bits per instruction
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # ═══════════════════════════════════════════════════════════════
        # REGISTER ANALYZER - Which registers look like counters?
        # ═══════════════════════════════════════════════════════════════
        self.reg_analyzer = nn.Sequential(
            nn.Linear(32, 64),  # 32 "counter likelihood" scores
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # ═══════════════════════════════════════════════════════════════
        # OUTPUT HEADS
        # ═══════════════════════════════════════════════════════════════
        self.type_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

        self.counter_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        self.iter_head = nn.Sequential(
            nn.Linear(64 + 1, 32),  # + selected counter value
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def compute_counter_likelihood(self, reg_values: torch.Tensor) -> torch.Tensor:
        """
        Score each register on how "counter-like" its value is.

        Good counter values: positive, in range [10, 100000]
        """
        # Score based on value range
        vals = reg_values.float()

        # Ideal range: 10 to 100000
        min_good, max_good = 10, 100000

        # Score: 1.0 for values in ideal range, decay outside
        in_range = (vals >= min_good) & (vals <= max_good)
        score = in_range.float()

        # Small penalty for very large values
        too_large = vals > max_good
        score = score - 0.5 * too_large.float()

        # Penalty for non-positive
        non_positive = vals <= 0
        score = score - 1.0 * non_positive.float()

        return score

    def forward(
        self,
        body_bits: torch.Tensor,    # [body_len, 32]
        reg_values: torch.Tensor,   # [32]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        body_len = body_bits.shape[0]

        # ═══════════════════════════════════════════════════════════════
        # EXTRACT OPCODES (bits 21-31)
        # ═══════════════════════════════════════════════════════════════
        opcodes = body_bits[:, 21:32]  # [body_len, 11]

        # Pad to max_body_len
        if body_len < self.max_body_len:
            padding = torch.zeros(self.max_body_len - body_len, 11, device=body_bits.device)
            opcodes = torch.cat([opcodes, padding], dim=0)

        opcode_flat = opcodes.flatten()  # [max_body_len * 11]
        opcode_features = self.opcode_embed(opcode_flat)  # [32]

        # ═══════════════════════════════════════════════════════════════
        # REGISTER ANALYSIS
        # ═══════════════════════════════════════════════════════════════
        counter_likelihood = self.compute_counter_likelihood(reg_values)  # [32]
        reg_features = self.reg_analyzer(counter_likelihood)  # [32]

        # ═══════════════════════════════════════════════════════════════
        # COMBINE AND PREDICT
        # ═══════════════════════════════════════════════════════════════
        combined = torch.cat([opcode_features, reg_features], dim=-1)  # [64]

        # Type prediction
        type_logits = self.type_head(combined)  # [4]

        # Counter prediction - bias toward registers with good values
        counter_logits = self.counter_head(combined)  # [32]
        counter_logits = counter_logits + counter_likelihood * 2  # Bias toward good values
        counter_probs = F.softmax(counter_logits, dim=-1)

        # Iteration prediction using selected counter's value
        best_counter = torch.argmax(counter_probs)
        counter_val = reg_values[best_counter].float()
        iter_input = torch.cat([combined, counter_val.unsqueeze(0) / 10000], dim=-1)
        log_iters = self.iter_head(iter_input)
        iterations = torch.pow(10.0, log_iters.clamp(1, 5))  # 10 to 100K

        return type_logits, counter_probs, iterations


class FastDataset(Dataset):
    """Fast dataset with pre-computed features."""

    def __init__(self, samples: List[Dict], max_body_len: int = 8):
        self.samples = samples
        self.max_body_len = max_body_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        instructions = sample['instructions']
        body_len = min(len(instructions), self.max_body_len)

        bits = torch.zeros(self.max_body_len, 32)
        for i, inst in enumerate(instructions[:body_len]):
            for j in range(32):
                bits[i, j] = float((inst >> j) & 1)

        regs = torch.tensor(sample['initial_regs'][:32], dtype=torch.float32)

        return (
            bits, regs, body_len,
            sample['loop_type'],
            sample['counter_reg'],
            sample['iterations']
        )


def train_fast(data_path: str = "loop_training_data.json", epochs: int = 50):
    """Fast training with lightweight model."""
    print("=" * 70)
    print("  FAST NEURAL LOOP TRAINING")
    print("=" * 70)
    print()

    with open(data_path, 'r') as f:
        samples = json.load(f)

    print(f"  Samples: {len(samples)}")

    model = FastLoopDetector().to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")
    print()

    dataset = FastDataset(samples)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    type_crit = nn.CrossEntropyLoss()
    reg_crit = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(epochs):
        model.train()
        type_correct = reg_correct = total = 0

        for batch in loader:
            bits, regs, body_lens, types, counter_regs, iterations = batch

            optimizer.zero_grad()
            batch_loss = 0

            for i in range(len(bits)):
                body_len = body_lens[i].item()
                body_bits = bits[i, :body_len].to(device)
                reg_vals = regs[i].to(device)

                type_logits, counter_probs, iter_pred = model(body_bits, reg_vals)

                type_target = torch.tensor([types[i]], device=device)
                reg_target = torch.tensor([counter_regs[i]], device=device)

                batch_loss += type_crit(type_logits.unsqueeze(0), type_target)
                batch_loss += reg_crit(counter_probs.unsqueeze(0), reg_target)

                if torch.argmax(type_logits).item() == types[i]:
                    type_correct += 1
                if torch.argmax(counter_probs).item() == counter_regs[i]:
                    reg_correct += 1
                total += 1

            batch_loss.backward()
            optimizer.step()

        type_acc = type_correct / total * 100
        reg_acc = reg_correct / total * 100
        combined = (type_acc + reg_acc) / 2

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: type={type_acc:.1f}%  reg={reg_acc:.1f}%")

        if combined > best_acc:
            best_acc = combined
            torch.save(model.state_dict(), "loop_detector_fast.pt")

    print()
    print(f"  Best combined accuracy: {best_acc:.1f}%")
    print(f"  Saved to: loop_detector_fast.pt")

    return model


def test_fast(model_path: str = "loop_detector_fast.pt", data_path: str = "loop_training_data.json"):
    """Test the fast model."""
    print("=" * 70)
    print("  TESTING FAST MODEL")
    print("=" * 70)
    print()

    model = FastLoopDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(data_path, 'r') as f:
        samples = json.load(f)

    type_correct = reg_correct = 0

    with torch.no_grad():
        for sample in samples:
            instructions = sample['instructions']
            body_len = len(instructions)

            bits = torch.zeros(body_len, 32, device=device)
            for i, inst in enumerate(instructions):
                for j in range(32):
                    bits[i, j] = float((inst >> j) & 1)

            regs = torch.tensor(sample['initial_regs'][:32], dtype=torch.float32, device=device)

            type_logits, counter_probs, _ = model(bits, regs)

            if torch.argmax(type_logits).item() == sample['loop_type']:
                type_correct += 1
            if torch.argmax(counter_probs).item() == sample['counter_reg']:
                reg_correct += 1

    n = len(samples)
    print(f"  Type accuracy: {type_correct/n*100:.1f}%")
    print(f"  Reg accuracy:  {reg_correct/n*100:.1f}%")


if __name__ == "__main__":
    print("Training fast loop detector...\n")
    train_fast(epochs=50)
    print()
    test_fast()
