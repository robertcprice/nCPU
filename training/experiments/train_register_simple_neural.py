#!/usr/bin/env python3
"""
SIMPLE TRULY NEURAL REGISTER FILE
==================================
Direct attention-weighted write. No complex gates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleNeuralRegisterFile(nn.Module):
    """Simple neural register: attention selects, outer product writes."""
    
    def __init__(self, n_regs=32, bit_width=64, key_dim=128):
        super().__init__()
        self.n_regs = n_regs
        self.bit_width = bit_width
        
        # Learned keys for each register
        self.register_keys = nn.Parameter(torch.randn(n_regs, key_dim) * 0.1)
        
        # Query encoder: 5-bit index → key
        self.query_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, key_dim),
        )
        
        # Temperature (learnable, starts sharp)
        self.temperature = nn.Parameter(torch.tensor(0.1))
    
    def _idx_to_bits(self, idx):
        B = idx.shape[0]
        bits = torch.zeros(B, 5, device=idx.device)
        for i in range(5):
            bits[:, i] = ((idx >> i) & 1).float()
        return bits
    
    def get_attention(self, idx):
        """Sharp attention for register selection."""
        idx_bits = self._idx_to_bits(idx)
        query = self.query_encoder(idx_bits)
        
        # Dot product attention
        similarity = torch.matmul(query, self.register_keys.T)
        
        temp = torch.clamp(self.temperature.abs(), min=0.01, max=1.0)
        attention = F.softmax(similarity / temp, dim=-1)
        return attention
    
    def read(self, memory, idx):
        """Neural read: attention @ memory."""
        attention = self.get_attention(idx)  # [B, n_regs]
        
        # XZR = 0
        is_xzr = (idx == 31).float().unsqueeze(-1)
        
        values = torch.matmul(attention, memory)  # [B, bit_width]
        values = values * (1 - is_xzr)
        return values
    
    def write(self, memory, idx, value):
        """
        Neural write: outer product update.
        new_memory = memory + attention^T @ value
        This ADDS the value weighted by attention.
        """
        attention = self.get_attention(idx)  # [B, n_regs]
        
        # XZR - don't write
        is_xzr = (idx == 31).float().unsqueeze(-1)
        attention = attention * (1 - is_xzr)
        
        # Outer product: [n_regs] x [bit_width] 
        # Average across batch
        w = attention.mean(dim=0)  # [n_regs]
        v = value.mean(dim=0)      # [bit_width]
        
        # Replace (not add): Use attention to blend
        # new[i] = (1 - w[i]) * old[i] + w[i] * v
        w_expanded = w.unsqueeze(-1)  # [n_regs, 1]
        v_expanded = v.unsqueeze(0)   # [1, bit_width]
        
        new_memory = (1 - w_expanded) * memory + w_expanded * v_expanded
        return new_memory


def train():
    print('=' * 70)
    print('🧠 SIMPLE NEURAL REGISTER FILE')
    print('   Direct attention-weighted write')
    print('=' * 70)
    
    model = SimpleNeuralRegisterFile().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Device: {device}')
    print(f'Parameters: {params:,}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    
    batch_size = 512
    num_epochs = 300
    best_acc = 0
    
    os.makedirs('models/final', exist_ok=True)
    
    print('\nTraining...')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        t0 = time.time()
        
        for batch_idx in range(100):
            optimizer.zero_grad()
            
            # Fresh memory
            memory = torch.zeros(32, 64, device=device)
            
            # Random register (not XZR)
            reg_idx = torch.randint(0, 31, (batch_size,), device=device)
            
            # Random value
            value = torch.rand(batch_size, 64, device=device)
            
            # WRITE
            memory = model.write(memory, reg_idx, value)
            
            # READ same register
            read_value = model.read(memory, reg_idx)
            
            # Loss: read should match write
            loss_rw = F.mse_loss(read_value, value)
            
            # Loss: attention should be sharp (cross-entropy)
            attention = model.get_attention(reg_idx)
            loss_att = F.cross_entropy(attention, reg_idx)
            
            loss = loss_rw * 5 + loss_att
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                diffs = (read_value - value).abs().max(dim=-1)[0]
                correct += (diffs < 0.15).sum().item()
                total += batch_size
        
        scheduler.step()
        
        acc = 100.0 * correct / total
        avg_loss = total_loss / 100
        elapsed = time.time() - t0
        
        if epoch % 10 == 0 or acc > best_acc:
            temp = model.temperature.item()
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f} RW={acc:.1f}% temp={temp:.3f} [{elapsed:.0f}s]', flush=True)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/final/neural_register_simple_best.pt')
            print(f'  ✅ Saved (acc={acc:.1f}%)')
        
        if acc >= 95.0:
            print('\n🎉 95%+ ACHIEVED!')
            break
    
    print(f'\nBest: {best_acc:.1f}%')
    
    # Test
    print('\n--- Test ---')
    model.eval()
    memory = torch.zeros(32, 64, device=device)
    with torch.no_grad():
        for r in [0, 5, 15, 30]:
            idx = torch.tensor([r], device=device)
            val = torch.rand(1, 64, device=device)
            memory = model.write(memory, idx, val)
            read = model.read(memory, idx)
            diff = (read - val).abs().max().item()
            att = model.get_attention(idx)
            print(f'X{r}: att_max={att.max():.3f} diff={diff:.4f} {"✅" if diff < 0.15 else "❌"}')


if __name__ == '__main__':
    train()
