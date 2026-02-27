#!/usr/bin/env python3
"""
NEURAL ELF LOADER - CURRICULUM + SUBNETWORK APPROACH
=====================================================
Same architecture that got decoder to 100%!

Key innovations:
1. CURRICULUM: Start simple, increase complexity
   - Phase 1: Just valid/invalid ELF detection
   - Phase 2: Add architecture detection (32/64, ARM/x86)
   - Phase 3: Add segment count (small range)
   - Phase 4: Add entry point (simplified range first)
   - Phase 5: Full range entry points

2. SUBNETWORKS: Separate specialized models
   - ValidatorNet: ELF magic detection
   - ArchNet: Architecture classification
   - SegmentNet: Segment count prediction
   - EntryNet: Entry point extraction

3. ENTRY POINT STRATEGY:
   - Entry points are typically 0x400000-0x800000 (4MB-8MB range)
   - This is only a 22-bit address space!
   - We'll predict the 22 significant bits, not all 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import random
import struct
import os
import time
import math

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Training on: {device}', flush=True)
if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name()}', flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# ============================================================
# SUBNETWORK 1: Header Validator
# ============================================================
class ValidatorNet(nn.Module):
    """Detects if bytes are valid ELF."""
    def __init__(self, d_model=128):
        super().__init__()
        # Focus on first 64 bytes (ELF header)
        self.embed = nn.Embedding(256, d_model)
        self.pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        self.attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        )

    def forward(self, header):
        # header: [B, 64] first 64 bytes
        x = self.embed(header) + self.pos
        x = self.attn(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# ============================================================
# SUBNETWORK 2: Architecture Classifier
# ============================================================
class ArchNet(nn.Module):
    """Classifies architecture: 32/64 bit, ARM64/x86."""
    def __init__(self, d_model=128):
        super().__init__()
        self.embed = nn.Embedding(256, d_model)
        self.pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        self.attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )

        # Key positions: byte 4 (class), bytes 18-19 (machine)
        self.is_64bit_head = nn.Linear(d_model, 2)
        self.is_arm64_head = nn.Linear(d_model, 2)

    def forward(self, header):
        x = self.embed(header) + self.pos
        x = self.attn(x)
        x = x.mean(dim=1)
        return {
            'is_64bit': self.is_64bit_head(x),
            'is_arm64': self.is_arm64_head(x)
        }


# ============================================================
# SUBNETWORK 3: Segment Counter
# ============================================================
class SegmentNet(nn.Module):
    """Counts program segments (0-15)."""
    def __init__(self, d_model=128):
        super().__init__()
        self.embed = nn.Embedding(256, d_model)
        self.pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        self.attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )

        self.classifier = nn.Linear(d_model, 16)

    def forward(self, header):
        x = self.embed(header) + self.pos
        x = self.attn(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# ============================================================
# SUBNETWORK 4: Entry Point Extractor (Byte-Level Classification)
# ============================================================
class EntryNet(nn.Module):
    """
    Extracts entry point using BYTE-LEVEL CLASSIFICATION.

    Key insight: Instead of predicting bits or using soft attention,
    we classify each of 8 bytes as one of 256 values.

    Why this works:
    - 8 independent 256-class problems (not 20 dependent binary decisions)
    - Each byte can be independently correct
    - Cross-entropy loss works naturally for classification
    - No floating point issues from soft attention

    Architecture:
    - Shared transformer encoder for context
    - 8 classification heads (one per byte)
    - Each head: context → 256 logits → predicted byte
    - Combine bytes with little-endian weights for final entry
    """
    def __init__(self, d_model=256, n_positions=64, n_entry_bytes=8):
        super().__init__()
        self.n_entry_bytes = n_entry_bytes

        self.embed = nn.Embedding(256, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_positions, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 8 byte classifiers - each predicts one byte value (0-255)
        self.byte_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, 256)  # 256 classes for byte value
            ) for _ in range(n_entry_bytes)
        ])

        # Little-endian combination weights
        self.register_buffer('le_weights',
            torch.tensor([256.0 ** i for i in range(8)]))

    def forward(self, header):
        """
        Extract entry point by classifying each byte.

        Args:
            header: [B, 64] raw bytes
        Returns:
            dict with 'entry' value and 'byte_logits' for training
        """
        x = self.embed(header) + self.pos
        x = self.transformer(x)
        ctx = x.mean(dim=1)  # [B, d_model]

        byte_logits = []
        predicted_bytes = []

        for head in self.byte_heads:
            logits = head(ctx)  # [B, 256]
            byte_logits.append(logits)
            # For inference: take argmax as predicted byte
            predicted_bytes.append(logits.argmax(dim=-1).float())  # [B]

        predicted_bytes = torch.stack(predicted_bytes, dim=-1)  # [B, 8]
        entry = (predicted_bytes * self.le_weights).sum(dim=-1)  # [B]

        return {
            'entry': entry,
            'byte_logits': torch.stack(byte_logits, dim=1)  # [B, 8, 256]
        }

    @staticmethod
    def get_entry_bytes(header_bytes, is_64bit):
        """Extract actual byte values at entry positions for training labels."""
        if is_64bit:
            # 64-bit: bytes 24-31
            return [header_bytes[24+i] for i in range(8)]
        else:
            # 32-bit: bytes 24-27, pad with zeros
            return [header_bytes[24+i] if i < 4 else 0 for i in range(8)]


# ============================================================
# Combined Model
# ============================================================
class CurriculumELFLoader(nn.Module):
    """Combined model with curriculum training support."""
    def __init__(self):
        super().__init__()
        self.validator = ValidatorNet()
        self.arch_net = ArchNet()
        self.segment_net = SegmentNet()
        self.entry_net = EntryNet()

    def forward(self, header, phase='full'):
        results = {}

        # Phase 1+: Always validate
        results['valid'] = self.validator(header)

        if phase in ['arch', 'segment', 'entry', 'full']:
            arch = self.arch_net(header)
            results['is_64bit'] = arch['is_64bit']
            results['is_arm64'] = arch['is_arm64']

        if phase in ['segment', 'entry', 'full']:
            results['segments'] = self.segment_net(header)

        if phase in ['entry', 'full']:
            entry = self.entry_net(header)
            results['entry'] = entry['entry']
            results['byte_logits'] = entry['byte_logits']  # [B, 8, 256]

        return results


# ============================================================
# Data Generation
# ============================================================
def generate_elf_header(difficulty='easy'):
    """Generate ELF header with controlled difficulty."""
    is_64bit = random.choice([True, False])
    is_arm64 = random.choice([True, False]) if is_64bit else False

    if is_64bit:
        header = bytearray(64)
        header[0:4] = b'\x7fELF'
        header[4] = 2  # 64-bit
        header[5] = 1  # Little endian
        header[6] = 1  # Version
        header[7] = 0  # ABI
        header[16:18] = struct.pack('<H', 2)  # ET_EXEC
        header[18:20] = struct.pack('<H', 183 if is_arm64 else 62)  # Machine
        header[20:24] = struct.pack('<I', 1)  # Version

        # Entry point based on difficulty
        if difficulty == 'easy':
            # Round numbers only
            entry = random.choice([0x400000, 0x500000, 0x600000, 0x700000, 0x800000])
        elif difficulty == 'medium':
            # Aligned addresses
            base = random.choice([0x400000, 0x500000, 0x600000, 0x700000])
            entry = base + random.randint(0, 15) * 0x10000
        else:
            # Any valid address
            entry = random.randint(0x400000, 0x8FFFFF)

        header[24:32] = struct.pack('<Q', entry)
        header[32:40] = struct.pack('<Q', 64)  # phoff

        num_segments = random.randint(1, 8)
        header[56:58] = struct.pack('<H', num_segments)
    else:
        header = bytearray(64)  # Pad to 64
        header[0:4] = b'\x7fELF'
        header[4] = 1  # 32-bit
        header[5] = 1
        header[6] = 1
        header[16:18] = struct.pack('<H', 2)
        header[18:20] = struct.pack('<H', 3)  # EM_386

        # 32-bit entry
        if difficulty == 'easy':
            entry = random.choice([0x8048000, 0x8049000, 0x804A000])
        else:
            entry = random.randint(0x8048000, 0x80FFFFF)
        # Mask to fit in our 22-bit range assumption
        entry = 0x400000 + (entry & 0xFFFFF)

        header[24:28] = struct.pack('<I', entry)

        num_segments = random.randint(1, 8)
        header[44:46] = struct.pack('<H', num_segments)

    # Entry bytes for byte-level classification training
    # Extract the actual byte values at positions 24-31 (or 24-27 for 32-bit)
    if is_64bit:
        entry_bytes = [header[24+i] for i in range(8)]
    else:
        entry_bytes = [header[24+i] if i < 4 else 0 for i in range(8)]

    return bytes(header), {
        'valid': 1,
        'is_64bit': 1 if is_64bit else 0,
        'is_arm64': 1 if is_arm64 else 0,
        'entry': entry,
        'entry_bytes': entry_bytes,  # [8] actual byte values for training
        'num_segments': num_segments,
    }


def generate_invalid_header():
    """Generate invalid ELF."""
    header = bytearray([random.randint(0, 255) for _ in range(64)])
    if header[0:4] == b'\x7fELF':
        header[0] = random.randint(0, 126)
    return bytes(header), {
        'valid': 0,
        'is_64bit': 0,
        'is_arm64': 0,
        'entry': 0x400000,
        'entry_bytes': [0] * 8,  # Placeholder for invalid
        'num_segments': 0,
    }


# ============================================================
# Training Loop
# ============================================================
def train():
    print('=' * 70, flush=True)
    print('🧠 NEURAL ELF LOADER - CURRICULUM + SUBNETWORKS', flush=True)
    print('   Same approach that got decoder to 100%!', flush=True)
    print('=' * 70, flush=True)

    batch_size = 4096  # H200 optimized

    # Curriculum phases - H200 optimized with more samples
    phases = [
        ('valid', 200000, 10, 'easy'),      # Phase 1: Just validation
        ('arch', 200000, 10, 'easy'),       # Phase 2: Add architecture
        ('segment', 200000, 10, 'easy'),    # Phase 3: Add segments
        ('entry', 500000, 30, 'easy'),      # Phase 4: Entry (easy)
        ('full', 500000, 30, 'medium'),     # Phase 5: Full (medium)
        ('full', 1000000, 50, 'hard'),      # Phase 6: Full (hard)
    ]

    model = CurriculumELFLoader().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scaler = GradScaler('cuda') if device == 'cuda' else None

    os.makedirs('models/final', exist_ok=True)
    best_acc = 0

    for phase_idx, (phase, num_samples, num_epochs, difficulty) in enumerate(phases):
        print(f'\n{"="*70}', flush=True)
        print(f'PHASE {phase_idx+1}: {phase.upper()} (difficulty={difficulty})', flush=True)
        print(f'{"="*70}', flush=True)

        # Generate data for this phase
        print(f'Generating {num_samples:,} samples...', flush=True)
        data = []
        for i in range(num_samples):
            if random.random() < 0.7:
                sample = generate_elf_header(difficulty)
            else:
                sample = generate_invalid_header()
            data.append(sample)

        # Adjust learning rate per phase
        for pg in optimizer.param_groups:
            pg['lr'] = 3e-4 * (0.5 ** phase_idx)

        for epoch in range(num_epochs):
            model.train()
            t0 = time.time()
            random.shuffle(data)

            total_loss = 0
            correct = {'valid': 0, '64bit': 0, 'arm64': 0, 'seg': 0, 'entry': 0}
            total = 0
            valid_count = 0

            num_batches = len(data) // batch_size

            for batch_idx in range(num_batches):
                batch = data[batch_idx * batch_size:(batch_idx + 1) * batch_size]

                headers = torch.tensor(
                    [[b for b in h[:64]] for h, _ in batch],
                    dtype=torch.long, device=device
                )

                valid_labels = torch.tensor([l['valid'] for _, l in batch], device=device)
                bit64_labels = torch.tensor([l['is_64bit'] for _, l in batch], device=device)
                arm64_labels = torch.tensor([l['is_arm64'] for _, l in batch], device=device)
                seg_labels = torch.tensor([l['num_segments'] for _, l in batch], device=device)

                # Entry point labels - byte values for classification
                entries = [l['entry'] for _, l in batch]
                entry_bytes = torch.tensor([l['entry_bytes'] for _, l in batch], device=device)  # [B, 8]

                optimizer.zero_grad()

                if scaler:
                    with autocast('cuda'):
                        outputs = model(headers, phase=phase)

                        loss = F.cross_entropy(outputs['valid'], valid_labels)

                        if 'is_64bit' in outputs:
                            loss = loss + F.cross_entropy(outputs['is_64bit'], bit64_labels)
                            loss = loss + F.cross_entropy(outputs['is_arm64'], arm64_labels)

                        if 'segments' in outputs:
                            loss = loss + F.cross_entropy(outputs['segments'], seg_labels)

                        if 'byte_logits' in outputs:
                            # Byte classification loss: classify each of 8 bytes
                            byte_logits = outputs['byte_logits']  # [B, 8, 256]
                            loss_entry = F.cross_entropy(
                                byte_logits.view(-1, 256),
                                entry_bytes.view(-1),
                                reduction='none'
                            ).view(batch_size, 8)
                            # Only count loss for valid ELF files
                            loss = loss + 2.0 * (loss_entry * valid_labels.unsqueeze(-1).float()).mean()

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(headers, phase=phase)
                    loss = F.cross_entropy(outputs['valid'], valid_labels)

                    if 'is_64bit' in outputs:
                        loss = loss + F.cross_entropy(outputs['is_64bit'], bit64_labels)
                        loss = loss + F.cross_entropy(outputs['is_arm64'], arm64_labels)

                    if 'segments' in outputs:
                        loss = loss + F.cross_entropy(outputs['segments'], seg_labels)

                    if 'byte_logits' in outputs:
                        byte_logits = outputs['byte_logits']
                        loss_entry = F.cross_entropy(
                            byte_logits.view(-1, 256),
                            entry_bytes.view(-1),
                            reduction='none'
                        ).view(batch_size, 8)
                        loss = loss + 2.0 * (loss_entry * valid_labels.unsqueeze(-1).float()).mean()

                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

                # Track accuracies
                with torch.no_grad():
                    correct['valid'] += (outputs['valid'].argmax(1) == valid_labels).sum().item()

                    if 'is_64bit' in outputs:
                        correct['64bit'] += (outputs['is_64bit'].argmax(1) == bit64_labels).sum().item()
                        correct['arm64'] += (outputs['is_arm64'].argmax(1) == arm64_labels).sum().item()

                    if 'segments' in outputs:
                        correct['seg'] += (outputs['segments'].argmax(1) == seg_labels).sum().item()

                    if 'byte_logits' in outputs:
                        # Compare predicted entry to actual entry
                        pred_entry = outputs['entry']  # [B]
                        for i in range(len(batch)):
                            if valid_labels[i] == 1:
                                valid_count += 1
                                # Check within small tolerance (floating point)
                                if abs(pred_entry[i].item() - entries[i]) < 1:
                                    correct['entry'] += 1

                    total += len(batch)

                if (batch_idx + 1) % 50 == 0:
                    print(f'  B{batch_idx+1}/{num_batches} Loss={loss.item():.4f}', flush=True)

            # Epoch stats
            avg_loss = total_loss / num_batches
            valid_acc = 100.0 * correct['valid'] / total

            acc_str = f'Valid={valid_acc:.1f}%'

            if 'is_64bit' in outputs:
                acc_str += f' 64bit={100.0*correct["64bit"]/total:.1f}%'
                acc_str += f' ARM64={100.0*correct["arm64"]/total:.1f}%'

            if 'segments' in outputs:
                acc_str += f' Seg={100.0*correct["seg"]/total:.1f}%'

            if 'byte_logits' in outputs:
                entry_acc = 100.0 * correct['entry'] / max(1, valid_count)
                acc_str += f' Entry={entry_acc:.1f}%'
            else:
                entry_acc = 0

            elapsed = time.time() - t0
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f} {acc_str} [{elapsed:.0f}s]', flush=True)

            # Early stopping for non-full phases: skip to next phase if 100%
            if phase != 'full':
                phase_metrics = [valid_acc]
                if 'is_64bit' in outputs:
                    phase_metrics.extend([100.0*correct['64bit']/total, 100.0*correct['arm64']/total])
                if 'segments' in outputs:
                    phase_metrics.append(100.0*correct['seg']/total)
                if 'byte_logits' in outputs:
                    phase_metrics.append(entry_acc)

                if all(m >= 99.9 for m in phase_metrics) and epoch >= 2:
                    print(f'  ⚡ Early stop: 100% achieved, moving to next phase!', flush=True)
                    break

            # Calculate combined accuracy
            if phase == 'full':
                combined = (valid_acc + 100.0*correct['64bit']/total +
                           100.0*correct['arm64']/total + 100.0*correct['seg']/total +
                           entry_acc) / 5.0

                if combined > best_acc:
                    best_acc = combined
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'combined_accuracy': combined / 100.0,
                        'valid_acc': valid_acc,
                        'entry_acc': entry_acc,
                        'phase': phase,
                        'epoch': epoch + 1,
                    }, 'models/final/neural_elf_loader_best.pt')
                    print(f'  ✅ Saved best (combined={combined:.1f}%)', flush=True)

                # Check for 100%
                if (valid_acc >= 100 and correct['64bit']/total >= 0.999 and
                    correct['arm64']/total >= 0.999 and correct['seg']/total >= 0.999 and
                    entry_acc >= 100):
                    print(f'\n🎉 ACHIEVED 100% ON ALL METRICS!', flush=True)
                    torch.save(model.state_dict(), 'models/final/neural_elf_loader_100pct.pt')
                    return

    print(f'\nBest combined accuracy: {best_acc:.1f}%', flush=True)
    print('Training complete!', flush=True)


if __name__ == '__main__':
    train()
