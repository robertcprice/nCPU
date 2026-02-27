#!/usr/bin/env python3
"""
TRULY NEURAL MMU v2
====================
Fixed approach: Learn a FIXED set of page mappings IN the weights.

The key insight: The neural register file works because it learns a
FIXED structure (32 registers). Same approach here - learn a fixed
page table with known mappings.

Training:
1. Pre-define a fixed page table (e.g., 1024 entries)
2. Store those mappings in network weights during training
3. Test: can the network recall the correct physical address?

This is like training a neural hash table!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


class TrulyNeuralMMUv2(nn.Module):
    """
    MMU with FIXED page table learned in weights.

    Like a neural hash table:
    - Fixed capacity (max_pages)
    - Each page has: virtual_page, physical_page, permissions
    - All stored IN the network weights
    """

    def __init__(self, max_pages=1024, page_bits=12, key_dim=64):
        super().__init__()
        self.max_pages = max_pages
        self.page_bits = page_bits  # 4KB pages
        self.key_dim = key_dim

        # We represent page numbers as 20-bit values (1M pages addressable)
        self.page_number_bits = 20

        # === PAGE TABLE IN WEIGHTS ===
        # Each entry: learned key for lookup
        self.page_keys = nn.Parameter(torch.randn(max_pages, key_dim) * 0.1)

        # Virtual page number for each slot (stored in weights)
        self.virtual_pages = nn.Parameter(torch.zeros(max_pages, self.page_number_bits))

        # Physical page number for each slot (stored in weights)
        self.physical_pages = nn.Parameter(torch.zeros(max_pages, self.page_number_bits))

        # Permissions: valid, r, w, x (stored in weights)
        self.permissions = nn.Parameter(torch.zeros(max_pages, 4))

        # Query encoder: page bits → key space
        self.query_encoder = nn.Sequential(
            nn.Linear(self.page_number_bits, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
        )

        # Temperature
        self.temperature = nn.Parameter(torch.tensor(0.5))

    def translate(self, virtual_page_bits):
        """
        Translate virtual page to physical page.

        Args:
            virtual_page_bits: [batch, page_number_bits]

        Returns:
            physical_page_bits: [batch, page_number_bits]
            valid: [batch]
            perms: [batch, 3] - r, w, x
        """
        batch = virtual_page_bits.shape[0]

        # Encode virtual page as query
        query = self.query_encoder(virtual_page_bits)  # [B, key_dim]

        # Compute attention over page table entries
        # Use similarity between query and stored virtual pages
        stored_virt = torch.sigmoid(self.virtual_pages)  # [max_pages, page_number_bits]

        # Also compute key-based similarity
        key_sim = torch.matmul(query, self.page_keys.T)  # [B, max_pages]

        # Combined similarity (page number match + key match)
        # We want entries where the virtual page matches
        page_sim = -((virtual_page_bits.unsqueeze(1) - stored_virt.unsqueeze(0)) ** 2).sum(dim=-1)

        combined_sim = key_sim + 2.0 * page_sim  # Weight page match higher

        # Temperature-scaled softmax
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(combined_sim / temp, dim=-1)  # [B, max_pages]

        # Read physical page via attention
        physical_raw = torch.sigmoid(self.physical_pages)
        physical_page_bits = torch.matmul(attention, physical_raw)

        # Read permissions
        perms_raw = torch.sigmoid(self.permissions)
        perms_all = torch.matmul(attention, perms_raw)
        valid = perms_all[:, 0]
        perms = perms_all[:, 1:]

        return physical_page_bits, valid, perms

    def populate(self, virtual_pages, physical_pages, permissions, slot_indices=None):
        """
        Populate page table entries.

        Args:
            virtual_pages: [n, page_number_bits]
            physical_pages: [n, page_number_bits]
            permissions: [n, 4]
            slot_indices: which slots to use (default: first n)
        """
        n = virtual_pages.shape[0]
        if slot_indices is None:
            slot_indices = torch.arange(n, device=virtual_pages.device)

        # Direct assignment to weights
        with torch.no_grad():
            # Convert bits to logits (inverse sigmoid)
            virt_logits = torch.log(virtual_pages.clamp(0.01, 0.99) / (1 - virtual_pages.clamp(0.01, 0.99)))
            phys_logits = torch.log(physical_pages.clamp(0.01, 0.99) / (1 - physical_pages.clamp(0.01, 0.99)))
            perm_logits = torch.log(permissions.clamp(0.01, 0.99) / (1 - permissions.clamp(0.01, 0.99)))

            self.virtual_pages.data[slot_indices] = virt_logits
            self.physical_pages.data[slot_indices] = phys_logits
            self.permissions.data[slot_indices] = perm_logits

            # Update keys to match virtual pages
            self.page_keys.data[slot_indices] = self.query_encoder(virtual_pages.float()).detach()


def int_to_bits(val, bits=20):
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def bits_to_int(bits_t):
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.tolist()))


def create_fixed_page_table(num_entries=512):
    """Create a fixed page table for training."""
    entries = []
    for i in range(num_entries):
        vpage = i * 4 + random.randint(0, 3)  # Some spacing
        ppage = vpage + random.randint(-10, 10)  # Near identity map
        ppage = max(0, ppage)

        entries.append({
            'vpage': vpage,
            'ppage': ppage,
            'valid': True,
            'r': random.random() < 0.95,
            'w': random.random() < 0.7,
            'x': random.random() < 0.3,
        })
    return entries


def generate_batch(page_table, batch_size, device):
    """Generate batch from fixed page table."""
    vpages = []
    ppages = []
    perms = []

    for _ in range(batch_size):
        entry = random.choice(page_table)
        vpages.append(int_to_bits(entry['vpage']))
        ppages.append(int_to_bits(entry['ppage']))
        perms.append([
            float(entry['valid']),
            float(entry['r']),
            float(entry['w']),
            float(entry['x'])
        ])

    return (
        torch.stack(vpages).to(device),
        torch.stack(ppages).to(device),
        torch.tensor(perms, device=device)
    )


def train():
    print("=" * 60)
    print("TRULY NEURAL MMU v2 TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("FIXED page table stored IN NETWORK WEIGHTS!")

    # Create fixed page table
    page_table = create_fixed_page_table(num_entries=512)
    print(f"Created {len(page_table)} page table entries")

    # Create model
    model = TrulyNeuralMMUv2(
        max_pages=1024,
        page_bits=12,
        key_dim=64
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Pre-populate the page table in weights
    print("Populating page table in weights...")
    vpages_init = torch.stack([int_to_bits(e['vpage']) for e in page_table]).to(device)
    ppages_init = torch.stack([int_to_bits(e['ppage']) for e in page_table]).to(device)
    perms_init = torch.tensor([
        [float(e['valid']), float(e['r']), float(e['w']), float(e['x'])]
        for e in page_table
    ], device=device)
    model.populate(vpages_init, ppages_init, perms_init)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 256

    for epoch in range(100):
        model.train()
        total_loss = 0
        t0 = time.time()

        for _ in range(100):
            vpages, ppages, perms = generate_batch(page_table, batch_size, device)

            optimizer.zero_grad()

            pred_phys, pred_valid, pred_perms = model.translate(vpages)

            loss_phys = F.mse_loss(pred_phys, ppages)
            loss_valid = F.binary_cross_entropy(pred_valid, perms[:, 0])
            loss_perms = F.binary_cross_entropy(pred_perms, perms[:, 1:])

            loss = loss_phys + loss_valid + 0.5 * loss_perms
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(20):
                vpages, ppages, perms = generate_batch(page_table, 100, device)
                pred_phys, pred_valid, _ = model.translate(vpages)

                # Check if physical page matches
                pred_int = [(bits_to_int(p.cpu()) for p in pred_phys)]
                true_int = [(bits_to_int(p.cpu()) for p in ppages)]

                for pred, true in zip(pred_phys, ppages):
                    pred_page = bits_to_int(pred.cpu())
                    true_page = bits_to_int(true.cpu())
                    if pred_page == true_page:
                        correct += 1
                    total += 1

        acc = correct / total
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} page_acc={100*acc:.1f}% [{elapsed:.1f}s]")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": acc,
                "op_name": "TRULY_NEURAL_MMU_V2",
                "num_entries": len(page_table)
            }, "models/final/truly_neural_mmu_v2_best.pt")
            print(f"  Saved (acc={100*acc:.1f}%)")

        if acc >= 0.95:
            print("95%+ ACCURACY!")
            break

    print(f"\nBest accuracy: {100*best_acc:.1f}%")

    # Final verification
    print("\nFinal verification:")
    model.eval()
    test_entries = page_table[:5]
    with torch.no_grad():
        for e in test_entries:
            vbits = int_to_bits(e['vpage']).unsqueeze(0).to(device)
            pbits = int_to_bits(e['ppage'])

            pred_phys, pred_valid, _ = model.translate(vbits)
            pred_page = bits_to_int(pred_phys[0].cpu())
            true_page = e['ppage']

            status = "OK" if pred_page == true_page else f"GOT {pred_page}"
            print(f"  vpage {e['vpage']} → ppage {true_page}: {status}")


if __name__ == "__main__":
    train()
