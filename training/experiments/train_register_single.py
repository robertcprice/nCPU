#!/usr/bin/env python3
"""
NEURAL REGISTER - Single item updates (no batch averaging)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralRegisterFile(nn.Module):
    def __init__(self, n_regs=32, bit_width=64, key_dim=128):
        super().__init__()
        self.n_regs = n_regs
        self.bit_width = bit_width
        
        # Learned keys
        self.register_keys = nn.Parameter(torch.randn(n_regs, key_dim) * 0.1)
        
        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(5, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, key_dim),
        )
        
        self.temperature = nn.Parameter(torch.tensor(0.1))
    
    def _idx_to_bits(self, idx):
        # idx is scalar or [1]
        if idx.dim() == 0:
            idx = idx.unsqueeze(0)
        bits = torch.zeros(1, 5, device=idx.device)
        for i in range(5):
            bits[0, i] = ((idx[0] >> i) & 1).float()
        return bits
    
    def get_attention(self, idx):
        idx_bits = self._idx_to_bits(idx)
        query = self.query_encoder(idx_bits)  # [1, key_dim]
        similarity = torch.matmul(query, self.register_keys.T)  # [1, n_regs]
        temp = torch.clamp(self.temperature.abs(), min=0.01)
        attention = F.softmax(similarity / temp, dim=-1)  # [1, n_regs]
        return attention.squeeze(0)  # [n_regs]
    
    def read(self, memory, idx):
        """Read: attention @ memory"""
        if idx.item() == 31:
            return torch.zeros(self.bit_width, device=memory.device)
        
        attention = self.get_attention(idx)  # [n_regs]
        value = torch.matmul(attention, memory)  # [bit_width]
        return value
    
    def write(self, memory, idx, value):
        """Write: outer product blend"""
        if idx.item() == 31:
            return memory
        
        attention = self.get_attention(idx)  # [n_regs]
        
        # new[i] = (1 - att[i]) * old[i] + att[i] * value
        att = attention.unsqueeze(-1)  # [n_regs, 1]
        val = value.unsqueeze(0)       # [1, bit_width]
        
        new_memory = (1 - att) * memory + att * val
        return new_memory


def train():
    print('=' * 70)
    print('🧠 NEURAL REGISTER - Single Item Updates')
    print('=' * 70)
    
    model = NeuralRegisterFile().to(device)
    print(f'Device: {device}')
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    num_epochs = 500
    ops_per_epoch = 5000
    best_acc = 0
    
    os.makedirs('models/final', exist_ok=True)
    
    print('\nTraining...')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        t0 = time.time()
        
        for _ in range(ops_per_epoch):
            optimizer.zero_grad()
            
            # Fresh memory
            memory = torch.zeros(32, 64, device=device)
            
            # Random register (not XZR)
            reg_idx = torch.randint(0, 31, (1,), device=device)
            
            # Random value
            value = torch.rand(64, device=device)
            
            # NEURAL WRITE
            memory = model.write(memory, reg_idx, value)
            
            # NEURAL READ
            read_value = model.read(memory, reg_idx)
            
            # Loss
            loss_rw = F.mse_loss(read_value, value)
            
            attention = model.get_attention(reg_idx)
            loss_att = F.cross_entropy(attention.unsqueeze(0), reg_idx)
            
            loss = loss_rw * 10 + loss_att
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                diff = (read_value - value).abs().max().item()
                if diff < 0.1:
                    correct += 1
        
        scheduler.step()
        
        acc = 100.0 * correct / ops_per_epoch
        avg_loss = total_loss / ops_per_epoch
        elapsed = time.time() - t0
        
        if epoch % 20 == 0 or acc > best_acc:
            temp = model.temperature.item()
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f} Acc={acc:.1f}% temp={temp:.3f} [{elapsed:.0f}s]', flush=True)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/final/neural_register_single_best.pt')
            print(f'  ✅ Saved (acc={acc:.1f}%)')
        
        if acc >= 95.0:
            print('\n🎉 95%+ ACHIEVED!')
            break
    
    print(f'\nBest: {best_acc:.1f}%')


if __name__ == '__main__':
    train()
