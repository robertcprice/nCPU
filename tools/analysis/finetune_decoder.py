#!/usr/bin/env python3
"""
Fine-tune Neural ARM64 Decoder on RTOS instructions.

The decoder learns to:
1. Extract rd, rn, rm registers (5-bit fields at known positions)
2. Classify operation TYPE (add, sub, and, or, load, store, branch, etc.)
3. Predict execution behavior from instruction bits

This is FULLY NEURAL - no hardcoded instruction handling!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
from pathlib import Path
from enum import IntEnum


# =============================================================================
# OPERATION CATEGORIES (what the neural ALU needs to know)
# =============================================================================

class OpCategory(IntEnum):
    """Categories that map to neural ALU operations."""
    ADD = 0       # Use neural ADD
    SUB = 1       # Use neural SUB
    AND = 2       # Use neural AND
    OR = 3        # Use neural OR
    XOR = 4       # Use neural XOR
    MUL = 5       # Multiplication
    DIV = 6       # Division
    SHIFT = 7     # Shifts (LSL, LSR, ASR)
    LOAD = 8      # Memory load
    STORE = 9     # Memory store
    BRANCH = 10   # Branch/jump
    COMPARE = 11  # Compare (sets flags)
    MOVE = 12     # Move/copy
    SYSTEM = 13   # System instructions
    UNKNOWN = 14  # For unsupported ops


# =============================================================================
# NEURAL DECODER ARCHITECTURE
# =============================================================================

class UniversalARM64Decoder(nn.Module):
    """
    Universal neural ARM64 decoder that learns from raw instruction bits.

    Instead of hardcoding instruction patterns, it learns:
    - Where register fields are (through attention)
    - What operation category the instruction belongs to
    - How to route to the appropriate neural ALU operation
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model

        # Bit embedding with strong positional encoding
        self.bit_embed = nn.Embedding(2, d_model // 2)
        self.pos_embed = nn.Embedding(32, d_model // 2)

        # Field position hints (helps learn ARM64 layout)
        # Bits 0-4: Rd, 5-9: Rn, 10-15: various, 16-20: Rm, 21-31: opcode
        self.field_hints = nn.Embedding(6, d_model // 4)  # 6 field types

        self.input_combine = nn.Linear(d_model + d_model // 4, d_model)

        # Transformer encoder for bit interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Register field extractors (attention-based)
        self.field_queries = nn.Parameter(torch.randn(4, d_model))  # rd, rn, rm, opcode
        self.field_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)

        # Register decoders with skip connections to raw bits
        self.rd_head = self._make_reg_decoder(d_model)
        self.rn_head = self._make_reg_decoder(d_model)
        self.rm_head = self._make_reg_decoder(d_model)

        # Direct bit connections for register extraction
        self.rd_bits = nn.Linear(5, 32)   # bits 0-4
        self.rn_bits = nn.Linear(5, 32)   # bits 5-9
        self.rm_bits = nn.Linear(5, 32)   # bits 16-20

        # Combine attention + direct
        self.rd_combine = nn.Linear(64, 32)
        self.rn_combine = nn.Linear(64, 32)
        self.rm_combine = nn.Linear(64, 32)

        # Operation category classifier
        self.cat_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, len(OpCategory))
        )

        # Memory operation detector (is_load, is_store)
        self.mem_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2)  # [is_load, is_store]
        )

        # Flag-setting detector
        self.flag_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)  # sets flags?
        )

        # Immediate value extraction (12-bit)
        self.imm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4096)  # 12-bit immediate
        )

    def _make_reg_decoder(self, d_model):
        return nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 32)
        )

    def forward(self, bits):
        """
        bits: [batch, 32] instruction bits
        Returns: dict with all decoded fields
        """
        batch = bits.shape[0]
        device = bits.device

        # Embed bits with position
        bit_idx = (bits > 0.5).long()
        pos_idx = torch.arange(32, device=device).unsqueeze(0).expand(batch, -1)

        # Field hints based on bit position
        field_idx = torch.zeros(32, dtype=torch.long, device=device)
        field_idx[0:5] = 0    # Rd field
        field_idx[5:10] = 1   # Rn field
        field_idx[10:16] = 2  # Immediate / other
        field_idx[16:21] = 3  # Rm field
        field_idx[21:24] = 4  # Opcode low
        field_idx[24:32] = 5  # Opcode high
        field_idx = field_idx.unsqueeze(0).expand(batch, -1)

        bit_emb = self.bit_embed(bit_idx)
        pos_emb = self.pos_embed(pos_idx)
        field_emb = self.field_hints(field_idx)

        x = torch.cat([bit_emb, pos_emb, field_emb], dim=-1)
        x = self.input_combine(x)

        # Self-attention over bits
        x = self.encoder(x)

        # Extract field representations with attention
        queries = self.field_queries.unsqueeze(0).expand(batch, -1, -1)
        fields, _ = self.field_attn(queries, x, x)

        # Decode registers using attention + direct bit connections
        rd_attn = self.rd_head(fields[:, 0])
        rd_direct = self.rd_bits(bits[:, 0:5])
        rd_logits = self.rd_combine(torch.cat([rd_attn, rd_direct], dim=-1))

        rn_attn = self.rn_head(fields[:, 1])
        rn_direct = self.rn_bits(bits[:, 5:10])
        rn_logits = self.rn_combine(torch.cat([rn_attn, rn_direct], dim=-1))

        rm_attn = self.rm_head(fields[:, 2])
        rm_direct = self.rm_bits(bits[:, 16:21])
        rm_logits = self.rm_combine(torch.cat([rm_attn, rm_direct], dim=-1))

        # Global context for classification
        global_ctx = fields[:, 3]  # Opcode field

        return {
            'rd': rd_logits,
            'rn': rn_logits,
            'rm': rm_logits,
            'category': self.cat_head(global_ctx),
            'mem_ops': self.mem_head(global_ctx),
            'sets_flags': self.flag_head(global_ctx),
            'immediate': self.imm_head(global_ctx),
        }


# =============================================================================
# INSTRUCTION LABELER (generates training labels from instructions)
# =============================================================================

def label_instruction(inst):
    """
    Analyze instruction and generate training labels.
    Returns: dict with rd, rn, rm, category, etc.
    """
    op = (inst >> 24) & 0xFF
    rd = inst & 0x1F
    rn = (inst >> 5) & 0x1F
    rm = (inst >> 16) & 0x1F
    imm12 = (inst >> 10) & 0xFFF

    # Determine category from opcode patterns
    # This is used ONLY for training labels, not execution!

    category = OpCategory.UNKNOWN
    is_load = False
    is_store = False
    sets_flags = False

    # ADD instructions
    if op in [0x91, 0x11, 0x8B, 0x0B]:
        category = OpCategory.ADD
    # SUB instructions
    elif op in [0xD1, 0x51, 0xCB, 0x4B]:
        category = OpCategory.SUB
    # AND instructions
    elif op in [0x8A, 0x0A, 0x12, 0x72]:
        category = OpCategory.AND
        if op == 0x72:
            sets_flags = True
    # OR instructions
    elif op in [0xAA, 0x2A, 0x32]:
        category = OpCategory.OR
    # XOR instructions
    elif op in [0xCA, 0x4A, 0x52]:
        category = OpCategory.XOR
    # MUL instructions
    elif op in [0x9B, 0x1B]:
        category = OpCategory.MUL
    # DIV instructions
    elif op in [0x9A, 0x1A]:
        category = OpCategory.DIV
    # Shifts
    elif op in [0xD3, 0x53]:
        category = OpCategory.SHIFT
    # Load/Store
    elif op in [0xF9, 0xB9, 0x39, 0x38, 0xB8, 0xA9, 0xA8, 0x29, 0x28, 0x79, 0x78]:
        if (inst >> 22) & 1:  # Check load bit for most
            category = OpCategory.LOAD
            is_load = True
        else:
            category = OpCategory.STORE
            is_store = True
    # Branch
    elif (op >> 2) == 0x05 or (op >> 2) == 0x25 or op == 0x54 or op in [0x34, 0xB4, 0x35, 0xB5]:
        category = OpCategory.BRANCH
    elif inst == 0xD65F03C0:  # RET
        category = OpCategory.BRANCH
    # Compare (sets flags)
    elif op in [0xF1, 0x71, 0xEB, 0x6B, 0x7A]:
        category = OpCategory.COMPARE
        sets_flags = True
    # Move
    elif op in [0xD2, 0x52, 0xF2, 0x72, 0x93]:
        category = OpCategory.MOVE
    # ADRP (address)
    elif (op & 0x9F) == 0x90 or (op & 0x9F) == 0x10:
        category = OpCategory.ADD  # Treat as ADD for address calculation
    # System
    elif inst == 0xD503201F:  # NOP
        category = OpCategory.SYSTEM

    return {
        'rd': rd,
        'rn': rn,
        'rm': rm,
        'category': category.value,
        'is_load': 1 if is_load else 0,
        'is_store': 1 if is_store else 0,
        'sets_flags': 1 if sets_flags else 0,
        'imm12': imm12,
    }


def inst_to_bits(inst):
    """Convert instruction to bit tensor."""
    return torch.tensor([float((inst >> i) & 1) for i in range(32)])


# =============================================================================
# TRAINING
# =============================================================================

def train_decoder(epochs=200, batch_size=128, device='cpu'):
    """Train the universal decoder on RTOS instructions."""

    print("="*70)
    print("UNIVERSAL NEURAL ARM64 DECODER - Training on RTOS")
    print("="*70)

    # Load RTOS instructions
    data_path = Path('rtos_instructions.pt')
    if not data_path.exists():
        print("ERROR: Run extract_rtos_instructions.py first!")
        return None

    raw_data = torch.load(data_path, weights_only=False)
    print(f"Loaded {len(raw_data)} instructions from RTOS")

    # Prepare training data
    bits_list = []
    labels = {
        'rd': [], 'rn': [], 'rm': [],
        'category': [], 'is_load': [], 'is_store': [],
        'sets_flags': [], 'imm12': []
    }

    for item in raw_data:
        inst = item['inst']
        bits_list.append(inst_to_bits(inst))

        label = label_instruction(inst)
        for k in labels:
            labels[k].append(label[k])

    bits = torch.stack(bits_list).to(device)
    for k in labels:
        labels[k] = torch.tensor(labels[k], device=device)

    # Model
    model = UniversalARM64Decoder(d_model=256).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # Try to load existing checkpoint
    ckpt_path = Path('models/final/universal_decoder.pt')
    if ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state, strict=False)
            print("Loaded existing checkpoint")
        except:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    print(f"\nTraining on {len(bits)} instructions...")
    print()

    best_acc = 0

    for epoch in range(epochs):
        model.train()

        # Shuffle
        perm = torch.randperm(len(bits))
        bits_shuffled = bits[perm]
        labels_shuffled = {k: v[perm] for k, v in labels.items()}

        total_loss = 0
        correct = {'rd': 0, 'rn': 0, 'rm': 0, 'cat': 0}
        n = 0

        for i in range(0, len(bits), batch_size):
            b = bits_shuffled[i:i+batch_size]
            t = {k: v[i:i+batch_size] for k, v in labels_shuffled.items()}

            out = model(b)

            # Losses
            loss_rd = ce(out['rd'], t['rd'])
            loss_rn = ce(out['rn'], t['rn'])
            loss_rm = ce(out['rm'], t['rm'])
            loss_cat = ce(out['category'], t['category'])
            loss_load = bce(out['mem_ops'][:, 0], t['is_load'].float())
            loss_store = bce(out['mem_ops'][:, 1], t['is_store'].float())
            loss_flags = bce(out['sets_flags'].squeeze(), t['sets_flags'].float())

            # Weight register losses higher (critical for correctness)
            loss = (loss_rd + loss_rn + loss_rm) * 3 + loss_cat + loss_load + loss_store + loss_flags

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            correct['rd'] += (out['rd'].argmax(1) == t['rd']).sum().item()
            correct['rn'] += (out['rn'].argmax(1) == t['rn']).sum().item()
            correct['rm'] += (out['rm'].argmax(1) == t['rm']).sum().item()
            correct['cat'] += (out['category'].argmax(1) == t['category']).sum().item()
            n += len(b)

        scheduler.step()

        accs = {k: v/n*100 for k, v in correct.items()}
        avg = sum(accs.values()) / len(accs)

        if epoch % 10 == 0 or avg > best_acc + 1:
            print(f"Ep {epoch:3d} | Loss: {total_loss:.3f} | "
                  f"Rd: {accs['rd']:.1f}% Rn: {accs['rn']:.1f}% Rm: {accs['rm']:.1f}% Cat: {accs['cat']:.1f}%")

        # Check for perfection
        if all(v >= 99.5 for v in accs.values()):
            print(f"\n✅ ACHIEVED 99.5%+ ON ALL METRICS!")
            break

        if avg > best_acc:
            best_acc = avg
            # Save checkpoint
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nBest average accuracy: {best_acc:.1f}%")
    print(f"Saved to {ckpt_path}")

    return model


if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = train_decoder(epochs=200, device=device)
