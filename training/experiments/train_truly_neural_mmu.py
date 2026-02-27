#!/usr/bin/env python3
"""
TRULY NEURAL MMU
=================
Page table mappings stored IN THE NETWORK WEIGHTS.

Like the Truly Neural Register File:
- Mappings stored as nn.Parameter tensors
- Read = attention-based lookup
- Write = Hebbian-style update of mappings
- NO hardcoded page tables - everything is neural!

This is a Neural Memory / Modern Hopfield Network for address translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


class TrulyNeuralMMU(nn.Module):
    """
    MMU where page table entries are stored IN the neural network weights.

    Architecture:
    - page_mappings: nn.Parameter [max_pages, 2 * page_bits] - virtual→physical stored IN weights
    - page_keys: nn.Parameter [max_pages, key_dim] - learned keys for attention lookup
    - permissions: nn.Parameter [max_pages, 4] - valid, r, w, x per page

    Read: virtual_addr → query → attention → physical_addr + perms
    Write: map new page → Hebbian update of page_mappings
    """

    def __init__(self, max_pages=4096, addr_bits=48, page_bits=12, key_dim=128):
        super().__init__()
        self.max_pages = max_pages
        self.addr_bits = addr_bits
        self.page_bits = page_bits
        self.page_number_bits = addr_bits - page_bits
        self.key_dim = key_dim

        # === PAGE MAPPINGS STORED IN WEIGHTS ===
        # Each entry: [virtual_page_bits, physical_page_bits]
        self.page_mappings = nn.Parameter(
            torch.zeros(max_pages, 2 * self.page_number_bits)
        )

        # Keys for attention-based lookup
        self.page_keys = nn.Parameter(
            torch.randn(max_pages, key_dim) * 0.1
        )

        # Permissions per page: [valid, r, w, x]
        self.page_permissions = nn.Parameter(
            torch.zeros(max_pages, 4)
        )

        # Query encoder: virtual page bits → key space
        self.query_encoder = nn.Sequential(
            nn.Linear(self.page_number_bits, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
        )

        # Temperature for attention sharpening
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Write learning rate
        self.write_lr = nn.Parameter(torch.tensor(0.1))

        # Value encoder for writes
        self.value_encoder = nn.Sequential(
            nn.Linear(2 * self.page_number_bits, 2 * self.page_number_bits),
            nn.GELU(),
            nn.Linear(2 * self.page_number_bits, 2 * self.page_number_bits),
        )

    def _get_attention(self, virtual_page_bits):
        """Get attention weights for page lookup."""
        batch = virtual_page_bits.shape[0]

        # Encode virtual page as query
        query = self.query_encoder(virtual_page_bits)  # [B, key_dim]

        # Compute attention over page keys
        similarity = torch.matmul(query, self.page_keys.T)  # [B, max_pages]

        # Temperature-scaled softmax
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(similarity / temp, dim=-1)  # [B, max_pages]

        return attention

    def translate(self, virtual_addr_bits):
        """
        Translate virtual address to physical address.

        Args:
            virtual_addr_bits: [batch, addr_bits] - virtual address as bits

        Returns:
            physical_bits: [batch, addr_bits] - physical address
            valid: [batch] - mapping valid
            perms: [batch, 3] - r, w, x
        """
        batch = virtual_addr_bits.shape[0]

        # Extract page number and offset
        offset_bits = virtual_addr_bits[:, :self.page_bits]
        page_bits = virtual_addr_bits[:, self.page_bits:]

        # Get attention for this virtual page
        attention = self._get_attention(page_bits)  # [B, max_pages]

        # Read physical page from mappings via attention
        # page_mappings: [max_pages, 2 * page_number_bits]
        # attention: [B, max_pages]
        mapping_values = self.page_mappings  # [max_pages, 2 * page_number_bits]

        # Weighted sum: [B, 2 * page_number_bits]
        read_mapping = torch.matmul(attention, mapping_values)

        # Extract physical page bits (second half of mapping)
        physical_page_bits = torch.sigmoid(read_mapping[:, self.page_number_bits:])

        # Read permissions
        perms_raw = torch.matmul(attention, self.page_permissions)  # [B, 4]
        perms = torch.sigmoid(perms_raw)
        valid = perms[:, 0]
        rwx = perms[:, 1:]

        # Combine physical page with offset
        physical_bits = torch.cat([offset_bits, physical_page_bits], dim=1)

        return physical_bits, valid, rwx

    def map_page(self, virtual_page_bits, physical_page_bits, permissions):
        """
        Map a virtual page to physical page (Hebbian update).

        Args:
            virtual_page_bits: [batch, page_number_bits] - virtual page
            physical_page_bits: [batch, page_number_bits] - physical page
            permissions: [batch, 4] - valid, r, w, x
        """
        batch = virtual_page_bits.shape[0]

        # Get attention for where to write
        attention = self._get_attention(virtual_page_bits)  # [B, max_pages]

        # Create new mapping value
        new_mapping = torch.cat([virtual_page_bits, physical_page_bits], dim=1)
        new_mapping = self.value_encoder(new_mapping)

        # Hebbian update: mappings += lr * attention.T @ new_mapping
        update = torch.matmul(attention.T, new_mapping)  # [max_pages, 2 * page_number_bits]

        # Apply update to weights
        lr = torch.clamp(self.write_lr.abs(), 0.01, 1.0)
        self.page_mappings.data = self.page_mappings.data + lr * update

        # Update permissions
        perm_update = torch.matmul(attention.T, permissions)  # [max_pages, 4]
        self.page_permissions.data = self.page_permissions.data + lr * perm_update


def addr_to_bits(addr, bits=48):
    return torch.tensor([(addr >> i) & 1 for i in range(bits)], dtype=torch.float32)


def bits_to_addr(bits_t):
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.tolist()))


def generate_batch(model, batch_size, device):
    """Generate training batch - write then read."""
    page_bits = model.page_number_bits
    addr_bits = model.addr_bits
    offset_bits = model.page_bits

    vaddrs = []
    paddrs = []
    perms_list = []

    for _ in range(batch_size):
        # Random pages
        vpage = random.randint(0, (1 << 20) - 1)  # Limit to 1M pages
        ppage = random.randint(0, (1 << 20) - 1)
        offset = random.randint(0, (1 << offset_bits) - 1)

        vaddr = (vpage << offset_bits) | offset
        paddr = (ppage << offset_bits) | offset

        perms = [1.0,  # valid
                 float(random.random() < 0.95),  # r
                 float(random.random() < 0.7),   # w
                 float(random.random() < 0.3)]   # x

        vaddrs.append(addr_to_bits(vaddr, addr_bits))
        paddrs.append(addr_to_bits(paddr, addr_bits))
        perms_list.append(perms)

    return (
        torch.stack(vaddrs).to(device),
        torch.stack(paddrs).to(device),
        torch.tensor(perms_list, device=device)
    )


def train():
    print("=" * 60)
    print("TRULY NEURAL MMU TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("Page mappings stored IN NETWORK WEIGHTS!")

    # Create model
    model = TrulyNeuralMMU(
        max_pages=4096,
        addr_bits=48,
        page_bits=12,
        key_dim=128
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 256

    for epoch in range(200):
        model.train()
        total_loss = 0
        t0 = time.time()

        for step in range(100):
            vaddrs, paddrs, perms = generate_batch(model, batch_size, device)

            # First, do some "writes" to populate the page table
            if step < 10 or random.random() < 0.1:
                # Write phase
                vpage_bits = vaddrs[:, model.page_bits:]
                ppage_bits = paddrs[:, model.page_bits:]
                model.map_page(vpage_bits, ppage_bits, perms)

            optimizer.zero_grad()

            # Translate
            pred_phys, pred_valid, pred_perms = model.translate(vaddrs)

            # Loss
            loss_phys = F.binary_cross_entropy(pred_phys, paddrs)
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
            for _ in range(10):
                vaddrs, paddrs, perms = generate_batch(model, 100, device)
                pred_phys, pred_valid, _ = model.translate(vaddrs)

                # Check page number accuracy
                pred_page = (pred_phys[:, model.page_bits:] > 0.5).float()
                true_page = (paddrs[:, model.page_bits:] > 0.5).float()
                page_match = (pred_page == true_page).all(dim=1)
                correct += page_match.sum().item()
                total += len(vaddrs)

        acc = correct / total
        elapsed = time.time() - t0

        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} page_acc={100*acc:.1f}% [{elapsed:.1f}s]")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": acc,
                "op_name": "TRULY_NEURAL_MMU",
                "architecture": "TrulyNeuralMMU"
            }, "models/final/truly_neural_mmu_best.pt")
            print(f"  Saved (acc={100*acc:.1f}%)")

        if acc >= 0.95:
            print("95%+ ACCURACY!")
            break

    print(f"\nBest accuracy: {100*best_acc:.1f}%")


if __name__ == "__main__":
    train()
