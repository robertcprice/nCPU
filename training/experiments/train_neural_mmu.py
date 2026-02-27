#!/usr/bin/env python3
"""
NEURAL MMU (Memory Management Unit)
====================================
Learns virtual→physical address translation.

Instead of hardcoded page tables, this neural network:
1. Takes virtual address as input
2. Outputs physical address + validity + permissions

Architecture: Transformer-based address translator
- Treats address bits as a sequence
- Learns page table structure from examples
- Outputs: physical_addr, valid, readable, writable, executable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


class NeuralMMU(nn.Module):
    """
    Neural Memory Management Unit.

    Learns to translate virtual addresses to physical addresses
    based on page table mappings it sees during training.
    """

    def __init__(self, addr_bits=48, page_bits=12, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.addr_bits = addr_bits
        self.page_bits = page_bits  # 4KB pages = 12 bits offset
        self.page_number_bits = addr_bits - page_bits

        # Embed each bit position
        self.bit_embed = nn.Embedding(2, d_model)  # 0 or 1
        self.pos_embed = nn.Parameter(torch.randn(1, addr_bits, d_model) * 0.02)

        # Transformer for understanding address structure
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.physical_addr_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, addr_bits)
        )

        self.valid_head = nn.Linear(d_model, 1)
        self.perm_head = nn.Linear(d_model, 3)  # r, w, x

    def forward(self, virtual_addr_bits):
        """
        Args:
            virtual_addr_bits: [batch, addr_bits] - virtual address as bits

        Returns:
            physical_bits: [batch, addr_bits] - physical address bits
            valid: [batch] - whether mapping is valid
            perms: [batch, 3] - r, w, x permissions
        """
        batch = virtual_addr_bits.shape[0]

        # Convert bits to embeddings
        bits_int = virtual_addr_bits.long()
        x = self.bit_embed(bits_int)  # [batch, addr_bits, d_model]
        x = x + self.pos_embed

        # Transform
        x = self.transformer(x)

        # Pool for classification heads (use mean)
        pooled = x.mean(dim=1)  # [batch, d_model]

        # Physical address (keep page offset same, translate page number)
        phys_logits = self.physical_addr_head(pooled)
        phys_page_bits = torch.sigmoid(phys_logits[:, self.page_bits:])
        # Concatenate: offset from virtual, page number from neural network
        phys_bits = torch.cat([
            virtual_addr_bits[:, :self.page_bits],  # Keep offset
            phys_page_bits  # Translate page number
        ], dim=1)

        # Validity and permissions
        valid = torch.sigmoid(self.valid_head(pooled).squeeze(-1))
        perms = torch.sigmoid(self.perm_head(pooled))

        return phys_bits, valid, perms


class NeuralTLB(nn.Module):
    """
    Neural Translation Lookaside Buffer.

    Fast path for recently used translations.
    Uses attention to match against cached entries.
    """

    def __init__(self, num_entries=64, addr_bits=48, d_model=64):
        super().__init__()
        self.num_entries = num_entries
        self.addr_bits = addr_bits

        # Learned TLB entries (virtual page → physical page mapping)
        self.entries = nn.Parameter(torch.randn(num_entries, addr_bits * 2))

        # Query network
        self.query_net = nn.Sequential(
            nn.Linear(addr_bits, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Key network for entries
        self.key_net = nn.Sequential(
            nn.Linear(addr_bits * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, virtual_addr_bits):
        """Look up in TLB, return hit/miss and physical addr if hit."""
        batch = virtual_addr_bits.shape[0]

        # Generate query
        q = self.query_net(virtual_addr_bits)  # [batch, d_model]

        # Generate keys for all entries
        k = self.key_net(self.entries)  # [num_entries, d_model]

        # Attention scores
        scores = torch.matmul(q, k.T) / math.sqrt(q.shape[-1])  # [batch, num_entries]
        attn = F.softmax(scores, dim=-1)

        # Get best matching entry
        best_idx = attn.argmax(dim=-1)  # [batch]
        confidence = attn.max(dim=-1).values  # [batch]

        # Extract physical address from best entry
        best_entries = self.entries[best_idx]  # [batch, addr_bits * 2]
        phys_bits = torch.sigmoid(best_entries[:, self.addr_bits:])

        # TLB hit if confidence > threshold
        hit = confidence > 0.9

        return phys_bits, hit, confidence


def addr_to_bits(addr, bits=48):
    """Convert integer address to bit tensor."""
    return torch.tensor([(addr >> i) & 1 for i in range(bits)], dtype=torch.float32)


def bits_to_addr(bits_t):
    """Convert bit tensor to integer address."""
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.tolist()))


def generate_page_table(num_mappings=1000, addr_bits=48, page_bits=12):
    """Generate synthetic page table mappings for training."""
    mappings = []
    page_size = 1 << page_bits
    max_page = 1 << (addr_bits - page_bits)

    for _ in range(num_mappings):
        # Random virtual page
        vpage = random.randint(0, min(max_page - 1, (1 << 20) - 1))  # Limit to 1M pages for training

        # Map to random physical page (or identity map with some offset)
        if random.random() < 0.5:
            # Identity-ish mapping
            ppage = vpage + random.randint(-100, 100)
            ppage = max(0, min(ppage, max_page - 1))
        else:
            # Random mapping
            ppage = random.randint(0, min(max_page - 1, (1 << 20) - 1))

        # Random permissions
        readable = random.random() < 0.95
        writable = random.random() < 0.7
        executable = random.random() < 0.3

        mappings.append({
            'vaddr': vpage * page_size,
            'paddr': ppage * page_size,
            'valid': True,
            'r': readable,
            'w': writable,
            'x': executable,
        })

    return mappings


def generate_batch(mappings, batch_size, addr_bits, device):
    """Generate training batch from page table mappings."""
    vaddrs = []
    paddrs = []
    valids = []
    perms = []

    for _ in range(batch_size):
        if random.random() < 0.9:
            # Valid mapping
            m = random.choice(mappings)
            # Add random offset within page
            offset = random.randint(0, (1 << 12) - 1)
            vaddr = m['vaddr'] + offset
            paddr = m['paddr'] + offset

            vaddrs.append(addr_to_bits(vaddr, addr_bits))
            paddrs.append(addr_to_bits(paddr, addr_bits))
            valids.append(1.0)
            perms.append([float(m['r']), float(m['w']), float(m['x'])])
        else:
            # Invalid mapping (unmapped address)
            vaddr = random.randint(1 << 30, 1 << 40)  # High address likely unmapped
            vaddrs.append(addr_to_bits(vaddr, addr_bits))
            paddrs.append(torch.zeros(addr_bits))
            valids.append(0.0)
            perms.append([0.0, 0.0, 0.0])

    return (
        torch.stack(vaddrs).to(device),
        torch.stack(paddrs).to(device),
        torch.tensor(valids, device=device),
        torch.tensor(perms, device=device)
    )


def train():
    print("=" * 60)
    print("NEURAL MMU TRAINING")
    print("=" * 60)
    print(f"Device: {device}")

    # Create model
    model = NeuralMMU(addr_bits=48, page_bits=12, d_model=128, nhead=8, num_layers=4).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Generate page table
    print("\nGenerating page table mappings...")
    mappings = generate_page_table(num_mappings=10000)
    print(f"Created {len(mappings)} mappings")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 256

    for epoch in range(200):
        model.train()
        total_loss = 0
        t0 = time.time()

        for _ in range(100):
            vaddrs, paddrs, valids, perms = generate_batch(mappings, batch_size, 48, device)

            optimizer.zero_grad()
            pred_phys, pred_valid, pred_perms = model(vaddrs)

            # Losses
            loss_phys = F.binary_cross_entropy(pred_phys, paddrs)
            loss_valid = F.binary_cross_entropy(pred_valid, valids)
            loss_perms = F.binary_cross_entropy(pred_perms, perms)

            loss = loss_phys + loss_valid + loss_perms
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        correct_addr = 0
        correct_valid = 0
        total = 0

        with torch.no_grad():
            for _ in range(10):
                vaddrs, paddrs, valids, perms = generate_batch(mappings, 100, 48, device)
                pred_phys, pred_valid, pred_perms = model(vaddrs)

                # Check address accuracy (page number must match)
                pred_pages = (pred_phys[:, 12:] > 0.5).float()
                true_pages = (paddrs[:, 12:] > 0.5).float()
                page_match = (pred_pages == true_pages).all(dim=1)
                correct_addr += (page_match & (valids > 0.5)).sum().item()

                # Check validity
                valid_match = ((pred_valid > 0.5) == (valids > 0.5))
                correct_valid += valid_match.sum().item()

                total += (valids > 0.5).sum().item()

        addr_acc = correct_addr / max(total, 1)
        valid_acc = correct_valid / 1000
        elapsed = time.time() - t0

        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} addr_acc={100*addr_acc:.1f}% valid_acc={100*valid_acc:.1f}% [{elapsed:.1f}s]")

        if addr_acc > best_acc:
            best_acc = addr_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": addr_acc,
                "op_name": "MMU",
                "architecture": "NeuralMMU"
            }, "models/final/neural_mmu_best.pt")
            print(f"  Saved (addr_acc={100*addr_acc:.1f}%)")

        if addr_acc >= 0.99:
            print("99%+ ACCURACY!")
            break

    print(f"\nBest accuracy: {100*best_acc:.1f}%")

    # Final verification
    print("\nFinal verification:")
    model.eval()
    test_mappings = mappings[:5]
    with torch.no_grad():
        for m in test_mappings:
            vaddr = m['vaddr']
            expected_paddr = m['paddr']

            vbits = addr_to_bits(vaddr, 48).unsqueeze(0).to(device)
            pred_phys, pred_valid, pred_perms = model(vbits)

            pred_paddr = bits_to_addr(pred_phys[0].cpu())
            # Only compare page numbers
            pred_page = pred_paddr >> 12
            exp_page = expected_paddr >> 12

            status = "OK" if pred_page == exp_page else f"GOT page {pred_page}"
            print(f"  0x{vaddr:012x} → 0x{expected_paddr:012x}: {status}")


if __name__ == "__main__":
    train()
