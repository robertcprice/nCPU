#!/usr/bin/env python3
"""
TRULY NEURAL REGISTER FILE
===========================
NO .data assignments! All operations are neural forward passes.
Uses Neural Turing Machine style differentiable memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrulyNeuralRegisterFile(nn.Module):
    """
    100% NEURAL register file.
    - READ: Attention-weighted retrieval from memory matrix
    - WRITE: Attention-weighted outer product update to memory
    No .data assignments anywhere!
    """
    
    def __init__(self, n_regs=32, bit_width=64, key_dim=256):
        super().__init__()
        self.n_regs = n_regs
        self.bit_width = bit_width
        
        # Register keys (for attention)
        self.register_keys = nn.Parameter(torch.randn(n_regs, key_dim))
        nn.init.orthogonal_(self.register_keys)
        
        # Query encoder: 5-bit → key_dim
        self.query_encoder = nn.Sequential(
            nn.Linear(5, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, key_dim),
        )
        
        # Value encoder: transforms written value
        self.value_encoder = nn.Sequential(
            nn.Linear(bit_width, bit_width * 2),
            nn.LayerNorm(bit_width * 2),
            nn.GELU(),
            nn.Linear(bit_width * 2, bit_width),
        )
        
        # Erase/write gates (like NTM)
        self.erase_gate = nn.Sequential(
            nn.Linear(bit_width + key_dim, 128),
            nn.GELU(),
            nn.Linear(128, bit_width),
            nn.Sigmoid(),
        )
        
        self.write_gate = nn.Sequential(
            nn.Linear(bit_width + key_dim, 128),
            nn.GELU(),
            nn.Linear(128, bit_width),
        )
        
        # Temperature
        self.temperature = nn.Parameter(torch.tensor(0.5))
    
    def _idx_to_bits(self, idx):
        B = idx.shape[0]
        bits = torch.zeros(B, 5, device=idx.device)
        for i in range(5):
            bits[:, i] = ((idx >> i) & 1).float()
        return bits
    
    def _get_attention(self, idx):
        idx_bits = self._idx_to_bits(idx)
        query = self.query_encoder(idx_bits)
        
        query_norm = F.normalize(query, dim=-1)
        keys_norm = F.normalize(self.register_keys, dim=-1)
        similarity = torch.matmul(query_norm, keys_norm.T)
        
        temp = torch.clamp(self.temperature.abs(), min=0.01)
        attention = F.softmax(similarity / temp, dim=-1)
        return attention, query
    
    def read(self, memory, idx):
        """
        NEURAL READ: attention-weighted retrieval.
        memory: [n_regs, bit_width]
        idx: [B]
        returns: [B, bit_width]
        """
        attention, _ = self._get_attention(idx)  # [B, n_regs]
        
        # XZR handling
        is_xzr = (idx == 31).float().unsqueeze(-1)
        
        # Weighted read
        values = torch.matmul(attention, memory)  # [B, bit_width]
        values = values * (1 - is_xzr)
        
        return values
    
    def write(self, memory, idx, value):
        """
        NEURAL WRITE: NTM-style erase-then-add.
        memory: [n_regs, bit_width]
        idx: [B]
        value: [B, bit_width]
        returns: new_memory [n_regs, bit_width]
        """
        attention, query = self._get_attention(idx)  # [B, n_regs], [B, key_dim]
        
        # XZR handling - don't write to X31
        is_xzr = (idx == 31).float().unsqueeze(-1)
        attention = attention * (1 - is_xzr)  # Zero out attention for XZR
        
        # Encode value
        encoded = self.value_encoder(value)  # [B, bit_width]
        
        # Compute erase and write vectors
        gate_input = torch.cat([encoded, query], dim=-1)  # [B, bit_width + key_dim]
        erase = self.erase_gate(gate_input)  # [B, bit_width]
        add = self.write_gate(gate_input)    # [B, bit_width]
        
        # NTM-style memory update (fully differentiable!)
        # For each batch item:
        # M_new = M * (1 - w ⊗ e) + w ⊗ a
        # where w is attention, e is erase, a is add
        
        # Average across batch for update
        w = attention.mean(dim=0)  # [n_regs]
        e = erase.mean(dim=0)      # [bit_width]
        a = add.mean(dim=0)        # [bit_width]
        
        # Outer products
        erase_matrix = torch.outer(w, e)  # [n_regs, bit_width]
        add_matrix = torch.outer(w, a)    # [n_regs, bit_width]
        
        # Update memory (NEURAL - no .data!)
        new_memory = memory * (1 - erase_matrix) + add_matrix
        
        return new_memory
    
    def forward(self, memory, write_idx, write_value, read_idx):
        """Full write-then-read cycle."""
        # Write
        new_memory = self.write(memory, write_idx, write_value)
        # Read from updated memory
        read_value = self.read(new_memory, read_idx)
        return read_value, new_memory


def train():
    print('=' * 70)
    print('🧠 TRULY NEURAL REGISTER FILE')
    print('   100% Neural READ and WRITE (NTM-style)')
    print('=' * 70)
    
    model = TrulyNeuralRegisterFile().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Device: {device}')
    print(f'Parameters: {params:,}')
    
    # Initial memory state
    memory = torch.zeros(32, 64, device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    batch_size = 512
    num_epochs = 500
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
            
            # Fresh memory each batch
            memory = torch.zeros(32, 64, device=device)
            
            # Random registers (not XZR)
            reg_indices = torch.randint(0, 31, (batch_size,), device=device)
            
            # Random values
            values = torch.rand(batch_size, 64, device=device)
            
            # Write then read SAME register
            read_values, new_memory = model(memory, reg_indices, values, reg_indices)
            
            # Loss 1: Read should match written
            loss_readwrite = F.mse_loss(read_values, values)
            
            # Loss 2: Attention accuracy
            attention, _ = model._get_attention(reg_indices)
            loss_attention = F.cross_entropy(attention, reg_indices)
            
            # Loss 3: Entropy (encourage sharp attention)
            entropy = -(attention * (attention + 1e-10).log()).sum(dim=-1).mean()
            loss_entropy = entropy * 0.1
            
            loss = loss_readwrite * 2 + loss_attention + loss_entropy
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            with torch.no_grad():
                diffs = (read_values - values).abs().max(dim=-1)[0]
                correct += (diffs < 0.15).sum().item()
                total += batch_size
        
        scheduler.step()
        
        acc = 100.0 * correct / total
        avg_loss = total_loss / num_batches
        elapsed = time.time() - t0
        
        if epoch % 20 == 0 or acc > best_acc:
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f} RW_Acc={acc:.1f}% [{elapsed:.0f}s]', flush=True)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/final/neural_register_truly_neural_best.pt')
            print(f'  ✅ Saved (acc={acc:.1f}%)')
        
        if acc >= 95.0:
            print('\n🎉 ACHIEVED 95%+ READ/WRITE ACCURACY!')
            break
    
    print(f'\nBest accuracy: {best_acc:.1f}%')
    
    # Final verification
    print('\n--- Final Verification ---')
    model.eval()
    memory = torch.zeros(32, 64, device=device)
    
    with torch.no_grad():
        for reg in [0, 5, 15, 30]:
            idx = torch.tensor([reg], device=device)
            val = torch.rand(1, 64, device=device)
            
            # Neural write
            memory = model.write(memory, idx, val)
            
            # Neural read
            read_val = model.read(memory, idx)
            
            diff = (read_val - val).abs().max().item()
            att, _ = model._get_attention(idx)
            att_idx = att.argmax().item()
            status = "✅" if diff < 0.15 else "❌"
            print(f'X{reg}: att→{att_idx} diff={diff:.4f} {status}')


if __name__ == '__main__':
    train()
