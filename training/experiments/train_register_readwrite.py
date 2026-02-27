#!/usr/bin/env python3
"""
Truly Neural Register File - FULL READ/WRITE Training
======================================================
Train the complete write-then-read cycle end-to-end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrulyNeuralRegisterFile(nn.Module):
    """Register file with values IN weights - trains READ and WRITE."""
    
    def __init__(self, n_regs=32, bit_width=64, key_dim=256):
        super().__init__()
        self.n_regs = n_regs
        self.bit_width = bit_width
        self.key_dim = key_dim
        
        # Register values ARE weights
        self.register_values = nn.Parameter(torch.randn(n_regs, bit_width) * 0.1)
        
        # Learned keys (orthogonal init)
        self.register_keys = nn.Parameter(torch.randn(n_regs, key_dim))
        nn.init.orthogonal_(self.register_keys)
        
        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(5, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, key_dim),
        )
        
        # Temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Value encoder for writes
        self.value_encoder = nn.Sequential(
            nn.Linear(bit_width, bit_width * 2),
            nn.LayerNorm(bit_width * 2),
            nn.GELU(),
            nn.Linear(bit_width * 2, bit_width),
        )
        
        # Write strength (learnable)
        self.write_strength = nn.Parameter(torch.tensor(1.0))
    
    def _idx_to_bits(self, idx):
        """Convert index to 5-bit binary."""
        B = idx.shape[0]
        bits = torch.zeros(B, 5, device=idx.device)
        for i in range(5):
            bits[:, i] = ((idx >> i) & 1).float()
        return bits
    
    def _get_attention(self, idx):
        """Get attention weights."""
        idx_bits = self._idx_to_bits(idx)
        query = self.query_encoder(idx_bits)
        
        # Cosine similarity
        query_norm = F.normalize(query, dim=-1)
        keys_norm = F.normalize(self.register_keys, dim=-1)
        similarity = torch.matmul(query_norm, keys_norm.T)
        
        temp = torch.clamp(self.temperature.abs(), min=0.01)
        attention = F.softmax(similarity / temp, dim=-1)
        return attention
    
    def read(self, idx):
        """Read using attention-weighted sum."""
        attention = self._get_attention(idx)
        values = torch.matmul(attention, self.register_values)
        
        # XZR = 0
        is_xzr = (idx == 31).float().unsqueeze(-1)
        values = values * (1 - is_xzr)
        return values
    
    def write_differentiable(self, idx, value):
        """Differentiable write - returns new register state."""
        is_xzr = (idx == 31).float().unsqueeze(-1)
        value = value * (1 - is_xzr)
        
        attention = self._get_attention(idx)  # [B, n_regs]
        encoded = self.value_encoder(value)   # [B, bit_width]
        
        # Soft write: blend old and new based on attention
        # new_regs[i] = (1 - att[i]) * old[i] + att[i] * encoded
        strength = torch.sigmoid(self.write_strength)
        
        # Expand for broadcasting
        att_expanded = attention.unsqueeze(-1)  # [B, n_regs, 1]
        encoded_expanded = encoded.unsqueeze(1)  # [B, 1, bit_width]
        old_regs = self.register_values.unsqueeze(0)  # [1, n_regs, bit_width]
        
        # Compute new state for each batch item
        # This is differentiable!
        new_regs = (1 - strength * att_expanded) * old_regs + strength * att_expanded * encoded_expanded
        
        # Average across batch (or take last)
        return new_regs.mean(dim=0)  # [n_regs, bit_width]
    
    def forward(self, write_idx, write_value, read_idx):
        """Full write-then-read cycle (differentiable)."""
        # Write
        new_regs = self.write_differentiable(write_idx, write_value)
        
        # Temporarily use new regs for read
        old_regs = self.register_values.data.clone()
        self.register_values.data = new_regs.detach()
        
        # Read
        read_value = self.read(read_idx)
        
        # Restore (gradients flow through new_regs)
        self.register_values.data = old_regs
        
        return read_value, new_regs


def train():
    print('=' * 70)
    print('🧠 TRULY NEURAL REGISTER FILE')
    print('   Training FULL READ/WRITE cycle')
    print('=' * 70)
    
    model = TrulyNeuralRegisterFile().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Device: {device}')
    print(f'Parameters: {params:,}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    
    batch_size = 1024
    num_epochs = 300
    num_batches = 100
    best_acc = 0
    
    os.makedirs('models/final', exist_ok=True)
    
    print('\nTraining...')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        t0 = time.time()
        
        for batch_idx in range(num_batches):
            optimizer.zero_grad()
            
            # Random registers (not XZR)
            reg_indices = torch.randint(0, 31, (batch_size,), device=device)
            
            # Random values to write
            values = torch.rand(batch_size, 64, device=device)
            
            # Write then read same register
            read_values, new_regs = model(reg_indices, values, reg_indices)
            
            # Loss 1: Read should match written value
            loss_readwrite = F.mse_loss(read_values, values)
            
            # Loss 2: Attention sharpness (cross-entropy)
            attention = model._get_attention(reg_indices)
            loss_attention = F.cross_entropy(attention, reg_indices)
            
            # Loss 3: Entropy regularization
            entropy = -(attention * (attention + 1e-10).log()).sum(dim=-1).mean()
            loss_entropy = entropy * 0.05
            
            loss = loss_readwrite + loss_attention * 0.5 + loss_entropy
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy: read matches write within tolerance
            with torch.no_grad():
                diffs = (read_values - values).abs().max(dim=-1)[0]
                correct += (diffs < 0.1).sum().item()
                total += batch_size
        
        scheduler.step()
        
        acc = 100.0 * correct / total
        avg_loss = total_loss / num_batches
        elapsed = time.time() - t0
        
        if epoch % 10 == 0 or acc > best_acc:
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f} ReadWrite={acc:.1f}% [{elapsed:.0f}s]', flush=True)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/final/neural_register_readwrite_best.pt')
            print(f'  ✅ Saved (acc={acc:.1f}%)')
        
        if acc >= 99.0:
            print('\n🎉 ACHIEVED 99%+ READ/WRITE ACCURACY!')
            break
    
    print(f'\nBest accuracy: {best_acc:.1f}%')
    
    # Final test
    print('\n--- Final Test ---')
    model.eval()
    with torch.no_grad():
        for reg in [0, 5, 15, 30]:
            idx = torch.tensor([reg], device=device)
            val = torch.rand(1, 64, device=device)
            
            # Write
            new_regs = model.write_differentiable(idx, val)
            model.register_values.data = new_regs
            
            # Read
            read_val = model.read(idx)
            
            diff = (read_val - val).abs().max().item()
            status = "✅" if diff < 0.1 else "❌"
            print(f'X{reg}: write→read diff={diff:.4f} {status}')


if __name__ == '__main__':
    train()
