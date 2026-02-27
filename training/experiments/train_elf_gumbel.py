#!/usr/bin/env python3
"""
ELF Loader with Gumbel-Softmax for comparison with Byte Classification.
Uses Gumbel-Softmax for differentiable discrete selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GumbelELFLoader(nn.Module):
    """ELF Loader using Gumbel-Softmax for entry point extraction."""
    
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        # Shared encoder for ELF header
        self.encoder = nn.Sequential(
            nn.Linear(64 * 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Validator head
        self.validator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Architecture detector (64-bit, ARM64)
        self.arch_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),  # [is_64bit, is_arm64]
        )
        
        # Segment counter
        self.segment_counter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 17),  # 0-16 segments
        )
        
        # Entry point: 8 bytes, each using Gumbel-Softmax over 64 positions
        # Position selector for each byte
        self.entry_position_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 64),  # 64 positions
            ) for _ in range(8)
        ])
        
    def forward(self, x, temperature=1.0):
        """
        Args:
            x: [B, 64, 8] - 64 bytes, 8 bits each
            temperature: Gumbel-Softmax temperature
        """
        B = x.shape[0]
        
        # Flatten and encode
        x_flat = x.view(B, -1)  # [B, 512]
        h = self.encoder(x_flat)  # [B, hidden]
        
        # Validator
        valid_logit = self.validator(h)  # [B, 1]
        
        # Architecture
        arch_logits = self.arch_detector(h)  # [B, 2]
        
        # Segments
        seg_logits = self.segment_counter(h)  # [B, 17]
        
        # Entry point using Gumbel-Softmax
        entry_bytes = []
        entry_positions = []
        
        for i, head in enumerate(self.entry_position_heads):
            pos_logits = head(h)  # [B, 64]
            
            # Gumbel-Softmax for differentiable position selection
            pos_soft = F.gumbel_softmax(pos_logits, tau=temperature, hard=False)  # [B, 64]
            entry_positions.append(pos_soft)
            
            # Extract byte value at selected position
            # x is [B, 64, 8], pos_soft is [B, 64]
            # weighted sum: [B, 64, 1] * [B, 64, 8] -> sum -> [B, 8]
            byte_value = (pos_soft.unsqueeze(-1) * x).sum(dim=1)  # [B, 8]
            entry_bytes.append(byte_value)
        
        entry_bytes = torch.stack(entry_bytes, dim=1)  # [B, 8, 8]
        entry_positions = torch.stack(entry_positions, dim=1)  # [B, 8, 64]
        
        return {
            'valid': valid_logit,
            'arch': arch_logits,
            'segments': seg_logits,
            'entry_bytes': entry_bytes,
            'entry_positions': entry_positions,
        }


def generate_elf_batch(batch_size, difficulty='easy'):
    """Generate batch of ELF-like data."""
    headers = torch.zeros(batch_size, 64, 8, device=device)
    targets = {
        'valid': torch.ones(batch_size, 1, device=device),
        'is_64bit': torch.ones(batch_size, device=device),
        'is_arm64': torch.ones(batch_size, device=device),
        'segments': torch.randint(1, 10, (batch_size,), device=device),
        'entry_positions': torch.zeros(batch_size, 8, dtype=torch.long, device=device),
        'entry_bytes': torch.zeros(batch_size, 8, 8, device=device),
    }
    
    for b in range(batch_size):
        # ELF magic
        headers[b, 0] = torch.tensor([0,1,1,1,1,1,1,1], dtype=torch.float32)  # 0x7F
        headers[b, 1] = torch.tensor([1,0,1,0,0,0,1,0], dtype=torch.float32)  # 'E'
        headers[b, 2] = torch.tensor([0,0,1,1,0,1,1,0], dtype=torch.float32)  # 'L'
        headers[b, 3] = torch.tensor([0,1,1,0,0,0,1,0], dtype=torch.float32)  # 'F'
        
        # Class (64-bit)
        headers[b, 4] = torch.tensor([0,1,0,0,0,0,0,0], dtype=torch.float32)
        
        # Machine (ARM64 = 0xB7 = 183)
        headers[b, 18] = torch.tensor([1,1,1,0,1,1,0,1], dtype=torch.float32)
        
        # Entry point at bytes 24-31
        if difficulty == 'easy':
            entry_pos = 24
        elif difficulty == 'medium':
            entry_pos = random.choice([24, 32, 40])
        else:
            entry_pos = random.randint(16, 48)
        
        # Generate entry point value
        entry_val = random.randint(0x400000, 0x800000)
        for i in range(8):
            byte_val = (entry_val >> (i * 8)) & 0xFF
            pos = entry_pos + i
            if pos < 64:
                bits = [(byte_val >> j) & 1 for j in range(8)]
                headers[b, pos] = torch.tensor(bits, dtype=torch.float32, device=device)
                targets['entry_positions'][b, i] = pos
                targets['entry_bytes'][b, i] = headers[b, pos]
        
        # Random noise in other positions
        for i in range(64):
            if i not in [0,1,2,3,4,18] and not (entry_pos <= i < entry_pos + 8):
                if random.random() < 0.3:
                    headers[b, i] = torch.rand(8, device=device)
    
    return headers, targets


def train():
    print('=' * 70)
    print('🎲 GUMBEL-SOFTMAX ELF LOADER')
    print('   Comparing with Byte Classification approach')
    print('=' * 70)
    
    model = GumbelELFLoader(hidden_dim=512).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    batch_size = 512
    num_epochs = 100
    best_acc = 0
    
    os.makedirs('models/final', exist_ok=True)
    
    print('\nTraining...')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_entry = 0
        total = 0
        t0 = time.time()
        
        # Curriculum: start easy, get harder
        if epoch < 30:
            difficulty = 'easy'
            temp = max(1.0, 5.0 - epoch * 0.15)
        elif epoch < 60:
            difficulty = 'medium'
            temp = max(0.5, 1.0 - (epoch - 30) * 0.02)
        else:
            difficulty = 'hard'
            temp = 0.5
        
        for batch_idx in range(100):
            optimizer.zero_grad()
            
            headers, targets = generate_elf_batch(batch_size, difficulty)
            outputs = model(headers, temperature=temp)
            
            # Losses
            loss_valid = F.binary_cross_entropy_with_logits(
                outputs['valid'], targets['valid']
            )
            
            loss_arch = F.binary_cross_entropy_with_logits(
                outputs['arch'][:, 0], targets['is_64bit']
            ) + F.binary_cross_entropy_with_logits(
                outputs['arch'][:, 1], targets['is_arm64']
            )
            
            loss_seg = F.cross_entropy(outputs['segments'], targets['segments'])
            
            # Entry point loss: position prediction
            loss_entry = 0
            for i in range(8):
                loss_entry += F.cross_entropy(
                    outputs['entry_positions'][:, i],  # [B, 64] logits from gumbel
                    targets['entry_positions'][:, i]   # [B] target positions
                )
            loss_entry /= 8
            
            # Also add byte reconstruction loss
            loss_bytes = F.mse_loss(outputs['entry_bytes'], targets['entry_bytes'])
            
            loss = loss_valid + loss_arch + loss_seg + loss_entry * 2 + loss_bytes
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Check entry accuracy
            with torch.no_grad():
                pred_pos = outputs['entry_positions'].argmax(dim=-1)  # [B, 8]
                correct = (pred_pos == targets['entry_positions']).all(dim=-1).sum().item()
                correct_entry += correct
                total += batch_size
        
        scheduler.step()
        
        acc = 100.0 * correct_entry / total
        avg_loss = total_loss / 100
        elapsed = time.time() - t0
        
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f} Entry={acc:.1f}% temp={temp:.2f} [{elapsed:.0f}s]', flush=True)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/final/gumbel_elf_loader_best.pt')
            print(f'  ✅ Saved best (entry={acc:.1f}%)')
        
        if acc >= 99.0:
            print('\n🎉 ACHIEVED 99%+ ENTRY ACCURACY!')
            break
    
    print(f'\nBest entry accuracy: {best_acc:.1f}%')
    print('Byte Classification achieved: 100%')
    print(f'Gumbel-Softmax achieved: {best_acc:.1f}%')


if __name__ == '__main__':
    train()
