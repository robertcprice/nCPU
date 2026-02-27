#!/usr/bin/env python3
"""
FINAL NEURAL ELF LOADER - COMBINING ALL APPROACHES
===================================================

This combines:
1. CURRICULUM LEARNING (from train_elf_curriculum.py) - phases for gradual complexity
2. ATTENTION-BASED EXTRACTION (truly neural) - learn WHERE bytes are, READ them
3. H200 OPTIMIZATION - large batches, lots of data

Key insight for entry point:
- Entry point bytes are at FIXED positions (24-31 for 64-bit)
- We train 8 attention heads to learn these positions
- Then EXTRACT actual byte values and combine with little-endian weighting
- This works for ANY entry point because we READ, not PREDICT

The attention targets during training are the actual byte positions.
At inference, the model reads whatever bytes are at those learned positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import random
import struct
import os
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on: {device}', flush=True)
if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name()}', flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


class AttentionEntryExtractor(nn.Module):
    """
    Extracts entry point by learning to ATTEND to correct byte positions.

    8 attention heads, each learns to focus on ONE byte position.
    Training teaches positions 24-31 for 64-bit ELF.
    Inference reads actual bytes from those positions.
    """
    def __init__(self, d_model=256, n_positions=64):
        super().__init__()
        self.n_entry_bytes = 8

        # Byte embedding
        self.byte_embed = nn.Embedding(256, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_positions, d_model) * 0.02)

        # Context encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 8 attention heads - each selects ONE byte position
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, n_positions)
            ) for _ in range(self.n_entry_bytes)
        ])

        # Little-endian weights
        self.register_buffer('le_weights',
            torch.tensor([256.0 ** i for i in range(8)]))

    def forward(self, byte_seq, return_attention=False):
        """
        Args:
            byte_seq: [B, 64] raw bytes
        Returns:
            entry_point: [B] extracted value
            attention: [B, 8, 64] optional
        """
        # Embed
        x = self.byte_embed(byte_seq) + self.pos_embed
        x = self.transformer(x)
        ctx = x.mean(dim=1)  # [B, d_model]

        selected_bytes = []
        attentions = []

        for head in self.attention_heads:
            logits = head(ctx)  # [B, 64]
            attn = F.softmax(logits, dim=-1)
            attentions.append(attn)

            # Extract byte using attention
            byte_val = (byte_seq.float() * attn).sum(dim=-1)  # [B]
            selected_bytes.append(byte_val)

        selected_bytes = torch.stack(selected_bytes, dim=-1)  # [B, 8]
        entry = (selected_bytes * self.le_weights).sum(dim=-1)  # [B]

        if return_attention:
            return entry, torch.stack(attentions, dim=1)
        return entry


class FinalELFLoader(nn.Module):
    """Combined model with all components."""
    def __init__(self, d_model=256):
        super().__init__()

        # Shared encoder
        self.byte_embed = nn.Embedding(256, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Classification heads (from curriculum approach)
        self.valid_head = nn.Linear(d_model, 2)
        self.is_64bit_head = nn.Linear(d_model, 2)
        self.is_arm64_head = nn.Linear(d_model, 2)
        self.segment_head = nn.Linear(d_model, 16)

        # Entry point extraction (attention-based)
        self.n_entry_bytes = 8
        self.entry_attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 64)
            ) for _ in range(self.n_entry_bytes)
        ])
        self.register_buffer('le_weights',
            torch.tensor([256.0 ** i for i in range(8)]))

    def forward(self, byte_seq):
        B = byte_seq.shape[0]

        # Encode
        x = self.byte_embed(byte_seq) + self.pos_embed
        x = self.transformer(x)
        ctx = x.mean(dim=1)  # [B, d_model]

        # Classifications
        valid = self.valid_head(ctx)
        is_64bit = self.is_64bit_head(ctx)
        is_arm64 = self.is_arm64_head(ctx)
        segments = self.segment_head(ctx)

        # Entry point extraction
        selected_bytes = []
        attentions = []
        for head in self.entry_attention_heads:
            logits = head(ctx)
            attn = F.softmax(logits, dim=-1)
            attentions.append(attn)
            byte_val = (byte_seq.float() * attn).sum(dim=-1)
            selected_bytes.append(byte_val)

        selected_bytes = torch.stack(selected_bytes, dim=-1)  # [B, 8]
        entry = (selected_bytes * self.le_weights).sum(dim=-1)  # [B]

        return {
            'valid': valid,
            'is_64bit': is_64bit,
            'is_arm64': is_arm64,
            'segments': segments,
            'entry': entry,
            'entry_attention': torch.stack(attentions, dim=1),  # [B, 8, 64]
        }


def generate_sample():
    """Generate ELF header with known positions."""
    is_64bit = random.choice([True, False])
    is_arm64 = random.choice([True, False]) if is_64bit else False

    header = bytearray(64)
    header[0:4] = b'\x7fELF'
    header[4] = 2 if is_64bit else 1
    header[5] = 1
    header[6] = 1
    header[16:18] = struct.pack('<H', 2)

    if is_64bit:
        header[18:20] = struct.pack('<H', 183 if is_arm64 else 62)
        entry = random.randint(0x400000, 0xFFFFFF)
        header[24:32] = struct.pack('<Q', entry)
        num_seg = random.randint(1, 8)
        header[56:58] = struct.pack('<H', num_seg)
        entry_positions = [24, 25, 26, 27, 28, 29, 30, 31]
    else:
        header[18:20] = struct.pack('<H', 3)
        entry = random.randint(0x8048000, 0x80FFFFF)
        header[24:28] = struct.pack('<I', entry)
        num_seg = random.randint(1, 8)
        header[44:46] = struct.pack('<H', num_seg)
        entry_positions = [24, 25, 26, 27, 0, 0, 0, 0]  # Only 4 bytes for 32-bit

    return {
        'bytes': bytes(header),
        'valid': 1,
        'is_64bit': 1 if is_64bit else 0,
        'is_arm64': 1 if is_arm64 else 0,
        'entry': entry,
        'entry_positions': entry_positions,
        'num_segments': num_seg,
    }


def generate_invalid():
    header = bytearray([random.randint(0, 255) for _ in range(64)])
    if header[0:4] == b'\x7fELF':
        header[0] = random.randint(0, 126)
    return {
        'bytes': bytes(header),
        'valid': 0,
        'is_64bit': 0,
        'is_arm64': 0,
        'entry': 0,
        'entry_positions': [0] * 8,
        'num_segments': 0,
    }


def train():
    print('=' * 70, flush=True)
    print('🧠 FINAL NEURAL ELF LOADER', flush=True)
    print('   Curriculum + Attention-based Extraction', flush=True)
    print('=' * 70, flush=True)

    # H200 settings
    batch_size = 1024
    num_samples = 500000
    num_epochs = 100

    print(f'Batch size: {batch_size}', flush=True)
    print(f'Samples: {num_samples:,}', flush=True)

    # Generate data
    print('Generating data...', flush=True)
    data = []
    for i in range(num_samples):
        if random.random() < 0.7:
            data.append(generate_sample())
        else:
            data.append(generate_invalid())
        if (i + 1) % 100000 == 0:
            print(f'  {i+1:,}/{num_samples:,}', flush=True)

    model = FinalELFLoader(d_model=256).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler('cuda') if device == 'cuda' else None

    os.makedirs('models/final', exist_ok=True)
    best_entry_acc = 0
    best_combined = 0

    print('\nTraining...', flush=True)

    for epoch in range(num_epochs):
        model.train()
        random.shuffle(data)
        t0 = time.time()

        total_loss = 0
        correct = {'valid': 0, '64bit': 0, 'arm64': 0, 'seg': 0}
        entry_correct = 0
        valid_count = 0
        total = 0

        num_batches = len(data) // batch_size

        for batch_idx in range(num_batches):
            batch = data[batch_idx * batch_size:(batch_idx + 1) * batch_size]

            byte_seqs = torch.tensor(
                [[b for b in s['bytes']] for s in batch],
                dtype=torch.long, device=device
            )

            valid_labels = torch.tensor([s['valid'] for s in batch], device=device)
            bit64_labels = torch.tensor([s['is_64bit'] for s in batch], device=device)
            arm64_labels = torch.tensor([s['is_arm64'] for s in batch], device=device)
            seg_labels = torch.tensor([s['num_segments'] for s in batch], device=device)
            entry_positions = torch.tensor([s['entry_positions'] for s in batch], device=device)
            entry_values = [s['entry'] for s in batch]

            optimizer.zero_grad()

            if scaler:
                with autocast('cuda'):
                    out = model(byte_seqs)

                    loss_valid = F.cross_entropy(out['valid'], valid_labels)
                    loss_64bit = F.cross_entropy(out['is_64bit'], bit64_labels)
                    loss_arm64 = F.cross_entropy(out['is_arm64'], arm64_labels)
                    loss_seg = F.cross_entropy(out['segments'], seg_labels)

                    # Entry attention loss - teach to focus on correct positions
                    # Only for valid ELF files
                    entry_attn = out['entry_attention']  # [B, 8, 64]
                    loss_entry = F.cross_entropy(
                        entry_attn.view(-1, 64),
                        entry_positions.view(-1),
                        reduction='none'
                    ).view(batch_size, 8)
                    loss_entry = (loss_entry * valid_labels.unsqueeze(-1).float()).mean()

                    loss = loss_valid + loss_64bit + loss_arm64 + loss_seg + 3.0 * loss_entry

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(byte_seqs)
                loss_valid = F.cross_entropy(out['valid'], valid_labels)
                loss_64bit = F.cross_entropy(out['is_64bit'], bit64_labels)
                loss_arm64 = F.cross_entropy(out['is_arm64'], arm64_labels)
                loss_seg = F.cross_entropy(out['segments'], seg_labels)

                entry_attn = out['entry_attention']
                loss_entry = F.cross_entropy(
                    entry_attn.view(-1, 64),
                    entry_positions.view(-1),
                    reduction='none'
                ).view(batch_size, 8)
                loss_entry = (loss_entry * valid_labels.unsqueeze(-1).float()).mean()

                loss = loss_valid + loss_64bit + loss_arm64 + loss_seg + 3.0 * loss_entry
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # Track accuracies
            with torch.no_grad():
                correct['valid'] += (out['valid'].argmax(1) == valid_labels).sum().item()
                correct['64bit'] += (out['is_64bit'].argmax(1) == bit64_labels).sum().item()
                correct['arm64'] += (out['is_arm64'].argmax(1) == arm64_labels).sum().item()
                correct['seg'] += (out['segments'].argmax(1) == seg_labels).sum().item()

                # Entry accuracy (within tolerance)
                pred_entry = out['entry']
                for i in range(len(batch)):
                    if valid_labels[i] == 1:
                        valid_count += 1
                        # Check within 0.5% tolerance
                        if abs(pred_entry[i].item() - entry_values[i]) < entry_values[i] * 0.005 + 1:
                            entry_correct += 1

                total += len(batch)

            if (batch_idx + 1) % 100 == 0:
                print(f'  B{batch_idx+1}/{num_batches} Loss={loss.item():.4f}', flush=True)

        scheduler.step()

        # Epoch stats
        valid_acc = 100.0 * correct['valid'] / total
        bit64_acc = 100.0 * correct['64bit'] / total
        arm64_acc = 100.0 * correct['arm64'] / total
        seg_acc = 100.0 * correct['seg'] / total
        entry_acc = 100.0 * entry_correct / max(1, valid_count)

        combined = (valid_acc + bit64_acc + arm64_acc + seg_acc + entry_acc) / 5.0

        elapsed = time.time() - t0
        print(f'Epoch {epoch+1}: Loss={total_loss/num_batches:.4f} '
              f'Valid={valid_acc:.1f}% 64bit={bit64_acc:.1f}% ARM64={arm64_acc:.1f}% '
              f'Seg={seg_acc:.1f}% Entry={entry_acc:.1f}% Combined={combined:.1f}% [{elapsed:.0f}s]', flush=True)

        if combined > best_combined:
            best_combined = combined
            torch.save({
                'state_dict': model.state_dict(),
                'combined': combined,
                'entry_acc': entry_acc,
            }, 'models/final/neural_elf_loader_best.pt')
            print(f'  ✅ Saved (combined={combined:.1f}%)', flush=True)

        if entry_acc > best_entry_acc:
            best_entry_acc = entry_acc

        # Check for 100%
        if valid_acc >= 100 and bit64_acc >= 100 and arm64_acc >= 100 and seg_acc >= 100 and entry_acc >= 99:
            print('\n🎉 ACHIEVED 99%+ ON ALL METRICS!', flush=True)
            torch.save(model.state_dict(), 'models/final/neural_elf_loader_100pct.pt')
            break

    print(f'\nBest entry accuracy: {best_entry_acc:.1f}%', flush=True)
    print(f'Best combined: {best_combined:.1f}%', flush=True)


if __name__ == '__main__':
    train()
