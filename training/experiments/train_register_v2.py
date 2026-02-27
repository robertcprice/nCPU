#!/usr/bin/env python3
"""
Truly Neural Register File v2 - Improved architecture.
Key insight: The query encoder needs more capacity to learn the 32-way classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrulyNeuralRegisterFileV2(nn.Module):
    """
    Register file where values are stored IN the network weights.
    V2: Bigger query encoder, learnable temperature, better initialization.
    """
    
    def __init__(self, n_regs=32, bit_width=64, key_dim=256):
        super().__init__()
        self.n_regs = n_regs
        self.bit_width = bit_width
        self.key_dim = key_dim
        
        # Register values ARE network weights - this is the core concept!
        self.register_values = nn.Parameter(torch.randn(n_regs, bit_width) * 0.1)
        
        # Learned keys for each register - orthogonal init for better separation
        self.register_keys = nn.Parameter(torch.randn(n_regs, key_dim))
        nn.init.orthogonal_(self.register_keys)
        
        # Query encoder: 5-bit index -> key_dim (BIGGER)
        self.query_encoder = nn.Sequential(
            nn.Linear(5, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, key_dim),
        )
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Value encoder for writes
        self.value_encoder = nn.Sequential(
            nn.Linear(bit_width, bit_width * 2),
            nn.GELU(),
            nn.Linear(bit_width * 2, bit_width),
        )
        
        self.write_lr = nn.Parameter(torch.tensor(0.1))
    
    def _idx_to_bits(self, idx):
        """Convert register index (0-31) to 5-bit binary."""
        B = idx.shape[0]
        bits = torch.zeros(B, 5, device=idx.device)
        for i in range(5):
            bits[:, i] = ((idx >> i) & 1).float()
        return bits
    
    def _get_attention(self, idx):
        """Get attention weights for register selection."""
        idx_bits = self._idx_to_bits(idx)  # [B, 5]
        query = self.query_encoder(idx_bits)  # [B, key_dim]
        
        # Normalize for cosine similarity
        query_norm = F.normalize(query, dim=-1)
        keys_norm = F.normalize(self.register_keys, dim=-1)
        
        # Cosine similarity
        similarity = torch.matmul(query_norm, keys_norm.T)  # [B, n_regs]
        
        # Temperature-scaled softmax (sharper as temp decreases)
        temp = torch.clamp(self.temperature.abs(), min=0.01)
        attention = F.softmax(similarity / temp, dim=-1)
        
        return attention
    
    def read(self, idx):
        """Read from register file."""
        attention = self._get_attention(idx)  # [B, n_regs]
        values = torch.matmul(attention, self.register_values)  # [B, bit_width]
        
        # XZR handling
        is_xzr = (idx == 31).float().unsqueeze(-1)
        values = values * (1 - is_xzr)
        
        return values
    
    def write(self, idx, value):
        """Write to register file (modifies weights)."""
        is_xzr = (idx == 31).float().unsqueeze(-1)
        value = value * (1 - is_xzr)
        
        attention = self._get_attention(idx)
        encoded_value = self.value_encoder(value)
        current = torch.matmul(attention, self.register_values)
        delta = encoded_value - current
        
        lr = torch.clamp(self.write_lr.abs(), min=0.01, max=1.0)
        update = lr * torch.matmul(attention.T, delta)
        
        with torch.no_grad():
            self.register_values.add_(update)
    
    def reset(self):
        """Reset register values."""
        with torch.no_grad():
            self.register_values.normal_(0, 0.1)


def train():
    print('=' * 70)
    print('🧠 TRULY NEURAL REGISTER FILE V2')
    print('   Improved architecture with bigger query encoder')
    print('=' * 70)
    
    model = TrulyNeuralRegisterFileV2().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Device: {device}')
    print(f'Parameters: {params:,}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    batch_size = 4096
    num_epochs = 200
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
            
            # Random register indices (not 31 = XZR)
            reg_indices = torch.randint(0, 31, (batch_size,), device=device)
            
            # Get attention weights
            attention = model._get_attention(reg_indices)  # [B, 32]
            
            # Loss 1: Cross-entropy (attention should peak at target)
            loss_ce = F.cross_entropy(attention, reg_indices)
            
            # Loss 2: Entropy regularization
            entropy = -(attention * (attention + 1e-10).log()).sum(dim=-1).mean()
            loss_entropy = entropy * 0.05
            
            # Loss 3: Contrastive - target attention should be much higher than others
            target_att = attention.gather(1, reg_indices.unsqueeze(1)).squeeze()  # [B]
            max_other = attention.clone()
            max_other.scatter_(1, reg_indices.unsqueeze(1), -1e9)
            max_other = max_other.max(dim=-1)[0]  # [B]
            margin_loss = F.relu(max_other - target_att + 0.5).mean()
            
            loss = loss_ce + loss_entropy + margin_loss
            
            # Accuracy
            with torch.no_grad():
                pred_idx = attention.argmax(dim=-1)
                correct += (pred_idx == reg_indices).sum().item()
                total += batch_size
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        acc = 100.0 * correct / total
        avg_loss = total_loss / num_batches
        temp = model.temperature.item()
        elapsed = time.time() - t0
        
        if epoch % 10 == 0 or acc > best_acc:
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f} Acc={acc:.1f}% temp={temp:.3f} [{elapsed:.0f}s]', flush=True)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/final/neural_register_v2_best.pt')
            print(f'  ✅ Saved best (acc={acc:.1f}%)')
        
        if acc >= 99.0:
            print('\n🎉 ACHIEVED 99%+ ATTENTION ACCURACY!')
            break
    
    print(f'\nBest accuracy: {best_acc:.1f}%')


if __name__ == '__main__':
    train()
