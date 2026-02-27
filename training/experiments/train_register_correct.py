#!/usr/bin/env python3
"""
NEURAL REGISTER - CORRECT Training
===================================
Must verify ATTENTION targets correct register, not just read==write!
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
        
        # Learned keys - initialize ORTHOGONALLY for separation
        self.register_keys = nn.Parameter(torch.randn(n_regs, key_dim))
        nn.init.orthogonal_(self.register_keys)
        
        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(5, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, key_dim),
        )
        
        self.temperature = nn.Parameter(torch.tensor(1.0))  # Start warm
    
    def _idx_to_bits(self, idx):
        if idx.dim() == 0:
            idx = idx.unsqueeze(0)
        bits = torch.zeros(1, 5, device=idx.device)
        for i in range(5):
            bits[0, i] = ((idx[0] >> i) & 1).float()
        return bits
    
    def get_attention(self, idx):
        idx_bits = self._idx_to_bits(idx)
        query = self.query_encoder(idx_bits)
        similarity = torch.matmul(query, self.register_keys.T)
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(similarity / temp, dim=-1)
        return attention.squeeze(0)


def train():
    print('=' * 70)
    print('🧠 NEURAL REGISTER - CORRECT Training')
    print('   Verifying attention targets CORRECT register!')
    print('=' * 70)
    
    model = NeuralRegisterFile().to(device)
    print(f'Device: {device}')
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    num_epochs = 500
    batch_size = 512
    best_acc = 0
    
    os.makedirs('models/final', exist_ok=True)
    
    print('\nTraining...')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_att = 0
        total = 0
        t0 = time.time()
        
        for _ in range(100):
            optimizer.zero_grad()
            
            # Random registers (not XZR)
            indices = torch.randint(0, 31, (batch_size,), device=device)
            
            # Compute attention for each
            loss_batch = 0
            for i in range(batch_size):
                idx = indices[i:i+1]
                attention = model.get_attention(idx)
                
                # Loss: Cross-entropy (attention should peak at target)
                loss_batch += F.cross_entropy(attention.unsqueeze(0), idx)
                
                # Check if attention argmax == target
                with torch.no_grad():
                    if attention.argmax().item() == idx.item():
                        correct_att += 1
                    total += 1
            
            loss = loss_batch / batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        acc = 100.0 * correct_att / total
        avg_loss = total_loss / 100
        temp = model.temperature.item()
        elapsed = time.time() - t0
        
        if epoch % 20 == 0 or acc > best_acc:
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f} AttAcc={acc:.1f}% temp={temp:.3f} [{elapsed:.0f}s]', flush=True)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/final/neural_register_correct_best.pt')
            print(f'  ✅ Saved (att_acc={acc:.1f}%)')
        
        if acc >= 99.0:
            print('\n🎉 99%+ ATTENTION ACCURACY!')
            break
    
    print(f'\nBest attention accuracy: {best_acc:.1f}%')
    
    # Verify
    print('\n--- Verification ---')
    model.eval()
    for r in [0, 5, 15, 30]:
        idx = torch.tensor([r], device=device)
        with torch.no_grad():
            att = model.get_attention(idx)
        max_idx = att.argmax().item()
        print(f'X{r}: attention → {max_idx} {"✅" if max_idx == r else "❌"}')


if __name__ == '__main__':
    train()
