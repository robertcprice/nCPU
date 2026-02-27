#!/usr/bin/env python3
"""
Neural Loop Detector V2 - Improved Architecture

Key improvements over V1:
1. Better register encoding: log-scale + presence flags
2. Attention mechanism to identify which instruction writes to counter
3. Log-scale iteration prediction
4. Instruction-register cross-attention

The network learns to identify:
- Which registers are "interesting" (contain loop-relevant values)
- Which instructions modify those registers
- What pattern the modifications follow (countdown, count-up, mem-fill)
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
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import math

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


class ImprovedLoopDetector(nn.Module):
    """
    Improved neural loop detector with better register understanding.

    Architecture:
    1. Instruction Encoder: Processes each instruction independently
    2. Register Encoder: Creates rich representation of register state
    3. Cross-Attention: Connects instructions to registers they reference
    4. Classifier Head: Predicts loop type, counter reg, iterations
    """

    def __init__(self, max_body_len: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.max_body_len = max_body_len
        self.hidden_dim = hidden_dim

        # ═══════════════════════════════════════════════════════════════
        # INSTRUCTION ENCODER
        # Learns ARM64 instruction structure
        # ═══════════════════════════════════════════════════════════════
        self.inst_embed = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Extract register indices from instruction bits
        # ARM64: Rd=bits[0:5], Rn=bits[5:10], Rm=bits[16:21]
        self.reg_field_extract = nn.Linear(32, 96)  # 3 regs * 32 one-hot

        # Sequence encoder for instruction patterns
        self.seq_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # ═══════════════════════════════════════════════════════════════
        # REGISTER ENCODER
        # Creates rich representation of register values
        # Key insight: encode RELATIVE magnitude, not absolute value
        # ═══════════════════════════════════════════════════════════════
        self.reg_embed = nn.Sequential(
            nn.Linear(64, hidden_dim),  # 32 log-values + 32 presence flags
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ═══════════════════════════════════════════════════════════════
        # CROSS-ATTENTION
        # Which instructions reference which registers?
        # ═══════════════════════════════════════════════════════════════
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional LSTM output
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # ═══════════════════════════════════════════════════════════════
        # OUTPUT HEADS
        # ═══════════════════════════════════════════════════════════════

        # Loop type classifier (none, count_up, countdown, mem_fill)
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 4)
        )

        # Counter register predictor
        # Uses attention over register values + instruction context
        self.counter_attn = nn.Linear(hidden_dim * 2, 32)

        # Iteration predictor (log-scale for better range)
        self.iter_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def encode_registers(self, reg_values: torch.Tensor) -> torch.Tensor:
        """
        Encode register values with log-scale + presence flags.

        Input: [32] raw register values
        Output: [64] encoded (32 log-values + 32 presence flags)
        """
        # Presence flags: which registers have non-zero values?
        presence = (reg_values != 0).float()

        # Log-scale values (with sign preservation)
        signs = torch.sign(reg_values)
        abs_vals = torch.abs(reg_values) + 1  # +1 to avoid log(0)
        log_vals = torch.log10(abs_vals.float()) * signs

        # Normalize log values to reasonable range
        log_vals = log_vals / 10.0  # log10(10B) ≈ 10

        return torch.cat([log_vals, presence], dim=-1)

    def forward(
        self,
        body_bits: torch.Tensor,    # [body_len, 32] instruction bits
        reg_values: torch.Tensor,   # [32] register values
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            loop_type_logits: [4] scores for each loop type
            counter_probs: [32] probability for each register being the counter
            iterations: [1] predicted iteration count (log-scale)
        """
        body_len = body_bits.shape[0]

        # ═══════════════════════════════════════════════════════════════
        # ENCODE INSTRUCTIONS
        # ═══════════════════════════════════════════════════════════════
        inst_embeds = self.inst_embed(body_bits)  # [body_len, hidden]

        # Get sequence encoding
        inst_embeds = inst_embeds.unsqueeze(0)  # [1, body_len, hidden]
        seq_out, (h_n, c_n) = self.seq_encoder(inst_embeds)  # [1, body_len, hidden*2]

        # Pool sequence: use last hidden states from both directions
        seq_summary = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [1, hidden*2]

        # ═══════════════════════════════════════════════════════════════
        # ENCODE REGISTERS
        # ═══════════════════════════════════════════════════════════════
        reg_encoded = self.encode_registers(reg_values)  # [64]
        reg_embed = self.reg_embed(reg_encoded)  # [hidden]

        # ═══════════════════════════════════════════════════════════════
        # COMBINE AND PREDICT
        # ═══════════════════════════════════════════════════════════════

        # Combined features
        seq_summary = seq_summary.squeeze(0)  # [hidden*2]
        combined = torch.cat([seq_summary, reg_embed], dim=-1)  # [hidden*3]

        # Loop type
        type_logits = self.type_head(combined)  # [4]

        # Counter register
        # Use attention: which registers are "interesting" given the instruction pattern?
        counter_scores = self.counter_attn(seq_summary)  # [32]

        # Weight by register presence (can't be counter if empty)
        presence = (reg_values != 0).float()
        counter_scores = counter_scores + (1 - presence) * -1e9  # Mask empty regs
        counter_probs = F.softmax(counter_scores, dim=-1)

        # Iteration count (log-scale)
        iter_input = torch.cat([combined, counter_probs], dim=-1)
        log_iters = self.iter_head(iter_input)  # [1]
        iterations = torch.pow(10.0, log_iters.clamp(-1, 6))  # 0.1 to 1M

        return type_logits, counter_probs, iterations


# ════════════════════════════════════════════════════════════════════════════════
# IMPROVED DATASET
# ════════════════════════════════════════════════════════════════════════════════

class ImprovedLoopDataset(Dataset):
    """Dataset with better preprocessing."""

    def __init__(self, samples: List[Dict], max_body_len: int = 8):
        self.samples = samples
        self.max_body_len = max_body_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # Convert instructions to bit tensor - PADDED to max_body_len
        instructions = sample['instructions']
        body_len = min(len(instructions), self.max_body_len)

        bits = torch.zeros(self.max_body_len, 32)  # Always same size for batching
        for i, inst in enumerate(instructions[:body_len]):
            for j in range(32):
                bits[i, j] = float((inst >> j) & 1)

        # Raw register values (encoding happens in model)
        regs = torch.tensor(sample['initial_regs'][:32], dtype=torch.float32)

        # Labels
        loop_type = sample['loop_type']
        counter_reg = sample['counter_reg']
        iterations = sample['iterations']

        return bits, regs, body_len, loop_type, counter_reg, iterations


# ════════════════════════════════════════════════════════════════════════════════
# IMPROVED TRAINING
# ════════════════════════════════════════════════════════════════════════════════

def train_improved(
    data_path: str = "loop_training_data.json",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 3e-4,
    save_path: str = "loop_detector_v2.pt"
):
    """Train the improved loop detector."""
    print("=" * 70)
    print("  IMPROVED NEURAL LOOP DETECTOR TRAINING (V2)")
    print("=" * 70)
    print()

    # Load data
    with open(data_path, 'r') as f:
        samples = json.load(f)

    print(f"  Training samples: {len(samples)}")

    # Create model
    model = ImprovedLoopDetector(max_body_len=8).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")
    print()

    # Create dataset
    dataset = ImprovedLoopDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss functions
    type_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        type_correct = 0
        reg_correct = 0
        total = 0

        for batch in loader:
            bits, regs, body_lens, types, counter_regs, iterations = batch

            optimizer.zero_grad()

            batch_loss = 0
            for i in range(len(bits)):
                body_len = body_lens[i].item()
                body_bits = bits[i, :body_len].to(device)
                reg_vals = regs[i].to(device)

                type_logits, counter_probs, iter_pred = model(body_bits, reg_vals)

                # Type loss
                type_target = types[i:i+1].to(device)
                type_loss = type_criterion(type_logits.unsqueeze(0), type_target)

                # Counter reg loss
                reg_target = counter_regs[i:i+1].to(device)
                reg_loss = reg_criterion(counter_probs.unsqueeze(0), reg_target)

                # Iteration loss (log-scale MSE)
                iter_target = torch.tensor([iterations[i]], dtype=torch.float32, device=device)
                log_pred = torch.log10(iter_pred.clamp(min=1))
                log_target = torch.log10(iter_target.clamp(min=1))
                iter_loss = F.mse_loss(log_pred, log_target)

                batch_loss += type_loss + reg_loss + 0.1 * iter_loss

                # Accuracy
                if torch.argmax(type_logits).item() == types[i].item():
                    type_correct += 1
                if torch.argmax(counter_probs).item() == counter_regs[i].item():
                    reg_correct += 1
                total += 1

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item()

        scheduler.step()

        avg_loss = total_loss / len(loader)
        type_acc = type_correct / total * 100
        reg_acc = reg_correct / total * 100
        combined_acc = (type_acc + reg_acc) / 2

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}  type={type_acc:.1f}%  reg={reg_acc:.1f}%  lr={scheduler.get_last_lr()[0]:.6f}")

        # Save best model (by combined accuracy)
        if combined_acc > best_acc:
            best_acc = combined_acc
            torch.save(model.state_dict(), save_path)

    print()
    print(f"  Training complete! Best combined accuracy: {best_acc:.1f}%")
    print(f"  Model saved to: {save_path}")

    return model


def test_improved(model_path: str = "loop_detector_v2.pt", data_path: str = "loop_training_data.json"):
    """Test the improved model."""
    print("=" * 70)
    print("  TESTING IMPROVED MODEL (V2)")
    print("=" * 70)
    print()

    model = ImprovedLoopDetector(max_body_len=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(data_path, 'r') as f:
        samples = json.load(f)

    type_correct = 0
    reg_correct = 0
    iter_errors = []

    with torch.no_grad():
        for sample in samples:
            instructions = sample['instructions']
            body_len = len(instructions)

            bits = torch.zeros(body_len, 32, device=device)
            for i, inst in enumerate(instructions):
                for j in range(32):
                    bits[i, j] = float((inst >> j) & 1)

            regs = torch.tensor(sample['initial_regs'][:32], dtype=torch.float32, device=device)

            type_logits, counter_probs, iter_pred = model(bits, regs)

            if torch.argmax(type_logits).item() == sample['loop_type']:
                type_correct += 1
            if torch.argmax(counter_probs).item() == sample['counter_reg']:
                reg_correct += 1

            iter_error = abs(iter_pred.item() - sample['iterations']) / max(1, sample['iterations'])
            iter_errors.append(iter_error)

    n = len(samples)
    print(f"  Samples: {n}")
    print(f"  Type accuracy:  {type_correct/n*100:.1f}%")
    print(f"  Reg accuracy:   {reg_correct/n*100:.1f}%")
    print(f"  Iter error:     {np.mean(iter_errors)*100:.1f}%")


# ════════════════════════════════════════════════════════════════════════════════
# BETTER SYNTHETIC DATA
# ════════════════════════════════════════════════════════════════════════════════

def generate_better_data(output_path: str = "loop_training_data_v2.json", num_samples: int = 2000):
    """Generate better training data with more variety."""
    print("=" * 70)
    print("  GENERATING IMPROVED TRAINING DATA")
    print("=" * 70)
    print()

    samples = []

    # ═══════════════════════════════════════════════════════════════
    # PATTERN 1: Simple countdown (SUB Rd, Rd, #imm)
    # ═══════════════════════════════════════════════════════════════
    for _ in range(num_samples // 4):
        counter_reg = np.random.randint(0, 28)
        decrement = np.random.choice([1, 2, 4, 8])
        iterations = np.random.randint(10, 10000)

        # SUB Xd, Xd, #imm
        sub_inst = 0xD1000000 | counter_reg | (counter_reg << 5) | ((decrement & 0xFFF) << 10)

        regs = [0] * 32
        regs[counter_reg] = iterations * decrement
        # Add some noise to other registers
        for r in range(32):
            if r != counter_reg:
                regs[r] = np.random.randint(0, 1000)

        samples.append({
            'instructions': [int(sub_inst)],
            'iterations': int(iterations),
            'counter_reg': int(counter_reg),
            'loop_type': 2,  # countdown
            'initial_regs': [int(r) for r in regs],
        })

    # ═══════════════════════════════════════════════════════════════
    # PATTERN 2: Count-up with ADD
    # ═══════════════════════════════════════════════════════════════
    for _ in range(num_samples // 4):
        counter_reg = np.random.randint(0, 28)
        limit_reg = (counter_reg + 1) % 28
        increment = np.random.choice([1, 2, 4, 8])
        start = np.random.randint(0, 100)
        iterations = np.random.randint(10, 1000)
        limit = start + iterations * increment

        # ADD Xd, Xd, #imm
        add_inst = 0x91000000 | counter_reg | (counter_reg << 5) | ((increment & 0xFFF) << 10)
        # CMP Xd, Xm
        cmp_inst = 0xEB00001F | (counter_reg << 5) | (limit_reg << 16)

        regs = [0] * 32
        regs[counter_reg] = start
        regs[limit_reg] = limit
        for r in range(32):
            if r not in [counter_reg, limit_reg]:
                regs[r] = np.random.randint(0, 1000)

        samples.append({
            'instructions': [int(add_inst), int(cmp_inst)],
            'iterations': int(iterations),
            'counter_reg': int(counter_reg),
            'loop_type': 1,  # count_up
            'initial_regs': [int(r) for r in regs],
        })

    # ═══════════════════════════════════════════════════════════════
    # PATTERN 3: Memory operations (STRB/STR + SUB)
    # ═══════════════════════════════════════════════════════════════
    for _ in range(num_samples // 4):
        ptr_reg = np.random.randint(0, 28)
        counter_reg = (ptr_reg + 1) % 28
        iterations = np.random.randint(10, 5000)

        # STRB/STR (simplified)
        store_inst = 0xB9000000 | (ptr_reg << 5)  # STR [Xn]
        # SUB counter
        sub_inst = 0xD1000400 | counter_reg | (counter_reg << 5)

        regs = [0] * 32
        regs[ptr_reg] = 0x10000
        regs[counter_reg] = iterations
        for r in range(32):
            if r not in [ptr_reg, counter_reg]:
                regs[r] = np.random.randint(0, 1000)

        samples.append({
            'instructions': [int(store_inst), int(sub_inst)],
            'iterations': int(iterations),
            'counter_reg': int(counter_reg),
            'loop_type': 3,  # mem_fill
            'initial_regs': [int(r) for r in regs],
        })

    # ═══════════════════════════════════════════════════════════════
    # PATTERN 4: Complex loops (multiple instructions)
    # ═══════════════════════════════════════════════════════════════
    for _ in range(num_samples // 4):
        counter_reg = np.random.randint(0, 28)
        iterations = np.random.randint(10, 5000)

        # Random "work" instruction
        work_reg = (counter_reg + 1) % 28
        work_inst = 0x91000400 | work_reg | (work_reg << 5)  # ADD Xwork, Xwork, #1

        # SUB counter, counter, #1
        sub_inst = 0xD1000400 | counter_reg | (counter_reg << 5)

        regs = [0] * 32
        regs[counter_reg] = iterations
        regs[work_reg] = 0
        for r in range(32):
            if r not in [counter_reg, work_reg]:
                regs[r] = np.random.randint(0, 1000)

        samples.append({
            'instructions': [int(work_inst), int(sub_inst)],
            'iterations': int(iterations),
            'counter_reg': int(counter_reg),
            'loop_type': 2,  # countdown (SUB controls loop)
            'initial_regs': [int(r) for r in regs],
        })

    np.random.shuffle(samples)

    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"  Generated {len(samples)} training samples")
    print(f"  Saved to: {output_path}")

    return samples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    if args.generate:
        generate_better_data()

    if args.train:
        train_improved(data_path="loop_training_data_v2.json", epochs=args.epochs)

    if args.test:
        test_improved()

    if not any([args.generate, args.train, args.test]):
        print("Running full V2 pipeline...\n")
        generate_better_data(num_samples=2000)
        print()
        train_improved(data_path="loop_training_data_v2.json", epochs=100)
        print()
        test_improved()


if __name__ == "__main__":
    main()
