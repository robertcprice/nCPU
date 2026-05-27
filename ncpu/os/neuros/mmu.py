"""Neural Memory Management Unit (MMU).

Replaces traditional page table walks with a trained neural network that
learns virtual→physical address mappings. The network uses page index
embeddings fed through an MLP to predict physical frame numbers.

Architecture:
    Input:  virtual page number (VPN) + address space ID (ASID)
    Embed:  learned embedding for page indices
    MLP:    embed → hidden → physical frame number
    Output: physical frame number (PFN) + permission bits

The MMU maintains a page table as ground truth for training and validation,
but all runtime translations go through the neural network.

Key design decisions:
    - 4KB pages (12-bit offset), matching ARM64 convention
    - 20-bit VPN space (1M virtual pages = 4GB virtual address space)
    - Physical frames allocated from a pool on GPU
    - Permission bits: R/W/X + valid + dirty + accessed
    - All tensors live on GPU — zero CPU-GPU sync during translation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path

from .device import default_device

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

PAGE_SIZE = 4096          # 4KB pages
PAGE_OFFSET_BITS = 12     # log2(4096)
VPN_BITS = 20             # 20-bit virtual page number → 1M pages
MAX_VIRTUAL_PAGES = 1 << VPN_BITS
MAX_ASID = 256            # 8-bit address space IDs

# Permission flags (bit positions)
PERM_VALID = 0
PERM_READ = 1
PERM_WRITE = 2
PERM_EXEC = 3
PERM_DIRTY = 4
PERM_ACCESSED = 5
NUM_PERM_BITS = 6


# ═══════════════════════════════════════════════════════════════════════════════
# Page Fault Exception
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PageFault:
    """Represents a page fault event."""
    virtual_addr: int
    vpn: int
    asid: int
    fault_type: str  # "not_mapped", "permission", "not_valid"


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Page Table Network
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralPageTable(nn.Module):
    """Neural network that learns virtual→physical page mappings.

    Architecture:
        VPN → Embedding(max_pages, embed_dim)
        ASID → Embedding(max_asid, asid_dim)
        [vpn_embed; asid_embed] → Linear → ReLU → Linear → ReLU → Linear
        Output: [pfn_logits(max_phys_frames), perm_bits(6)]

    For small page tables (< 1K entries), this is essentially a learned
    lookup table. For larger mappings, the network must generalize over
    the embedding space.
    """

    def __init__(self, max_virtual_pages: int = 4096,
                 max_physical_frames: int = 4096,
                 embed_dim: int = 64, hidden_dim: int = 256,
                 asid_dim: int = 16, max_asid: int = MAX_ASID):
        super().__init__()
        self.max_virtual_pages = max_virtual_pages
        self.max_physical_frames = max_physical_frames

        self.vpn_embed = nn.Embedding(max_virtual_pages, embed_dim)
        self.asid_embed = nn.Embedding(max_asid, asid_dim)

        input_dim = embed_dim + asid_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_physical_frames + NUM_PERM_BITS),
        )

    def forward(self, vpn: torch.Tensor, asid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Translate virtual page numbers to physical frame numbers.

        Args:
            vpn:  [batch] int64 — virtual page numbers
            asid: [batch] int64 — address space IDs

        Returns:
            pfn:   [batch] int64 — predicted physical frame numbers
            perms: [batch, 6] float — permission bits (sigmoid-activated)
        """
        vpn_e = self.vpn_embed(vpn)
        asid_e = self.asid_embed(asid)
        x = torch.cat([vpn_e, asid_e], dim=-1)
        out = self.mlp(x)

        pfn_logits = out[:, :self.max_physical_frames]
        perm_logits = out[:, self.max_physical_frames:]

        pfn = pfn_logits.argmax(dim=-1)
        perms = torch.sigmoid(perm_logits)

        return pfn, perms


# ═══════════════════════════════════════════════════════════════════════════════
# Neural MMU
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralMMU:
    """Neural Memory Management Unit.

    Manages virtual→physical address translation using a trained neural network.
    Maintains a conventional page table as ground truth for training/validation,
    and a physical frame allocator for page allocation.

    All state lives on GPU as tensors. Translation is a single forward pass
    through the neural network — no page table walks.
    """

    def __init__(self, max_virtual_pages: int = 4096,
                 max_physical_frames: int = 4096,
                 device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.max_virtual_pages = max_virtual_pages
        self.max_physical_frames = max_physical_frames

        # Neural page table network
        self.net = NeuralPageTable(
            max_virtual_pages=max_virtual_pages,
            max_physical_frames=max_physical_frames,
        ).to(self.device)

        # Ground truth page table: vpn → (pfn, perms)
        # -1 means unmapped
        self.page_table_pfn = torch.full(
            (MAX_ASID, max_virtual_pages), -1,
            dtype=torch.int64, device=self.device
        )
        self.page_table_perms = torch.zeros(
            MAX_ASID, max_virtual_pages, NUM_PERM_BITS,
            dtype=torch.float32, device=self.device
        )

        # Physical frame allocator — bitmap
        self.frame_bitmap = torch.zeros(
            max_physical_frames, dtype=torch.bool, device=self.device
        )
        self.next_free_frame = 0

        # Statistics
        self.translations = 0
        self.page_faults = 0
        self._trained = False

    # ─── Frame Allocation ─────────────────────────────────────────────────

    def alloc_frame(self) -> int:
        """Allocate a physical frame. Returns frame number or -1 if OOM."""
        # Find first free frame
        free = (~self.frame_bitmap).nonzero(as_tuple=True)[0]
        if len(free) == 0:
            return -1
        frame = int(free[0].item())
        self.frame_bitmap[frame] = True
        return frame

    def free_frame(self, pfn: int):
        """Free a physical frame."""
        if 0 <= pfn < self.max_physical_frames:
            self.frame_bitmap[pfn] = False

    @property
    def free_frames(self) -> int:
        """Number of available physical frames."""
        return int((~self.frame_bitmap).sum().item())

    # ─── Page Table Management ────────────────────────────────────────────

    def map_page(self, vpn: int, pfn: int, asid: int = 0,
                 read: bool = True, write: bool = False,
                 execute: bool = False) -> bool:
        """Map a virtual page to a physical frame in the ground truth table.

        Args:
            vpn: Virtual page number
            pfn: Physical frame number
            asid: Address space ID
            read/write/execute: Permission flags

        Returns:
            True if mapping was created successfully
        """
        if vpn >= self.max_virtual_pages or pfn >= self.max_physical_frames:
            return False
        if asid >= MAX_ASID:
            return False

        self.page_table_pfn[asid, vpn] = pfn
        self.page_table_perms[asid, vpn, PERM_VALID] = 1.0
        self.page_table_perms[asid, vpn, PERM_READ] = float(read)
        self.page_table_perms[asid, vpn, PERM_WRITE] = float(write)
        self.page_table_perms[asid, vpn, PERM_EXEC] = float(execute)
        self.page_table_perms[asid, vpn, PERM_DIRTY] = 0.0
        self.page_table_perms[asid, vpn, PERM_ACCESSED] = 0.0
        return True

    def unmap_page(self, vpn: int, asid: int = 0):
        """Unmap a virtual page."""
        if vpn < self.max_virtual_pages and asid < MAX_ASID:
            old_pfn = int(self.page_table_pfn[asid, vpn].item())
            self.page_table_pfn[asid, vpn] = -1
            self.page_table_perms[asid, vpn] = 0.0
            if old_pfn >= 0:
                self.free_frame(old_pfn)

    def alloc_and_map(self, vpn: int, asid: int = 0,
                      read: bool = True, write: bool = False,
                      execute: bool = False) -> int:
        """Allocate a frame and map it to the given virtual page.

        Returns:
            Physical frame number, or -1 if allocation failed
        """
        pfn = self.alloc_frame()
        if pfn < 0:
            return -1
        self.map_page(vpn, pfn, asid, read, write, execute)
        return pfn

    # ─── Translation ──────────────────────────────────────────────────────

    def translate(self, virtual_addr: int, asid: int = 0,
                  write: bool = False, execute: bool = False) -> Tuple[int, Optional[PageFault]]:
        """Translate a virtual address to a physical address.

        Uses the neural network if trained, otherwise falls back to
        the ground truth page table.

        Args:
            virtual_addr: Virtual address to translate
            asid: Address space ID
            write: Whether this is a write access
            execute: Whether this is an instruction fetch

        Returns:
            (physical_addr, page_fault) — page_fault is None on success
        """
        vpn = virtual_addr >> PAGE_OFFSET_BITS
        offset = virtual_addr & (PAGE_SIZE - 1)
        self.translations += 1

        if vpn >= self.max_virtual_pages:
            self.page_faults += 1
            return -1, PageFault(virtual_addr, vpn, asid, "not_mapped")

        if self._trained:
            return self._translate_neural(vpn, offset, asid, write, execute)
        return self._translate_table(vpn, offset, asid, write, execute)

    def translate_batch(self, virtual_addrs: torch.Tensor,
                        asid: int = 0) -> torch.Tensor:
        """Batch translate virtual addresses to physical addresses.

        Args:
            virtual_addrs: [batch] int64 — virtual addresses
            asid: Address space ID (same for all)

        Returns:
            [batch] int64 — physical addresses (-1 for faults)
        """
        vpns = virtual_addrs >> PAGE_OFFSET_BITS
        offsets = virtual_addrs & (PAGE_SIZE - 1)

        if self._trained:
            asids = torch.full_like(vpns, asid)
            with torch.no_grad():
                pfns, perms = self.net(vpns, asids)
            valid = perms[:, PERM_VALID] > 0.5
            phys = pfns * PAGE_SIZE + offsets
            phys[~valid] = -1
            return phys

        # Fallback: table lookup
        pfns = self.page_table_pfn[asid, vpns]
        valid = pfns >= 0
        phys = torch.where(valid, pfns * PAGE_SIZE + offsets, torch.tensor(-1, device=self.device))
        return phys

    def _translate_neural(self, vpn: int, offset: int, asid: int,
                          write: bool, execute: bool) -> Tuple[int, Optional[PageFault]]:
        """Translate using the neural network."""
        vpn_t = torch.tensor([vpn], dtype=torch.int64, device=self.device)
        asid_t = torch.tensor([asid], dtype=torch.int64, device=self.device)

        with torch.no_grad():
            pfn, perms = self.net(vpn_t, asid_t)

        pfn_val = int(pfn[0].item())
        perm_vals = perms[0]

        # Check validity
        if perm_vals[PERM_VALID] < 0.5:
            self.page_faults += 1
            return -1, PageFault(vpn * PAGE_SIZE + offset, vpn, asid, "not_valid")

        # Check permissions
        if perm_vals[PERM_READ] < 0.5:
            self.page_faults += 1
            return -1, PageFault(vpn * PAGE_SIZE + offset, vpn, asid, "permission")

        if write and perm_vals[PERM_WRITE] < 0.5:
            self.page_faults += 1
            return -1, PageFault(vpn * PAGE_SIZE + offset, vpn, asid, "permission")

        if execute and perm_vals[PERM_EXEC] < 0.5:
            self.page_faults += 1
            return -1, PageFault(vpn * PAGE_SIZE + offset, vpn, asid, "permission")

        return pfn_val * PAGE_SIZE + offset, None

    def _translate_table(self, vpn: int, offset: int, asid: int,
                         write: bool, execute: bool) -> Tuple[int, Optional[PageFault]]:
        """Translate using the ground truth page table (fallback)."""
        pfn = int(self.page_table_pfn[asid, vpn].item())

        if pfn < 0:
            self.page_faults += 1
            return -1, PageFault(vpn * PAGE_SIZE + offset, vpn, asid, "not_mapped")

        perms = self.page_table_perms[asid, vpn]

        if perms[PERM_VALID] < 0.5:
            self.page_faults += 1
            return -1, PageFault(vpn * PAGE_SIZE + offset, vpn, asid, "not_valid")

        if write and perms[PERM_WRITE] < 0.5:
            self.page_faults += 1
            return -1, PageFault(vpn * PAGE_SIZE + offset, vpn, asid, "permission")

        if execute and perms[PERM_EXEC] < 0.5:
            self.page_faults += 1
            return -1, PageFault(vpn * PAGE_SIZE + offset, vpn, asid, "permission")

        # Mark accessed (and dirty if write)
        self.page_table_perms[asid, vpn, PERM_ACCESSED] = 1.0
        if write:
            self.page_table_perms[asid, vpn, PERM_DIRTY] = 1.0

        return pfn * PAGE_SIZE + offset, None

    # ─── Training ─────────────────────────────────────────────────────────

    def train_from_table(self, epochs: int = 100, lr: float = 1e-3,
                         asid: int = 0) -> Dict:
        """Train the neural page table from ground truth mappings.

        Extracts all valid mappings from the page table and trains the
        network to reproduce them.

        Returns:
            Training statistics dict
        """
        # Gather valid mappings
        valid_mask = self.page_table_pfn[asid] >= 0
        valid_vpns = valid_mask.nonzero(as_tuple=True)[0]

        if len(valid_vpns) == 0:
            logger.warning("No valid mappings to train on")
            return {"error": "no_mappings"}

        target_pfns = self.page_table_pfn[asid, valid_vpns]
        target_perms = self.page_table_perms[asid, valid_vpns]
        asids = torch.full_like(valid_vpns, asid)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        pfn_loss_fn = nn.CrossEntropyLoss()
        perm_loss_fn = nn.BCEWithLogitsLoss()

        best_acc = 0.0
        for epoch in range(epochs):
            self.net.train()
            optimizer.zero_grad()

            vpn_e = self.net.vpn_embed(valid_vpns)
            asid_e = self.net.asid_embed(asids)
            x = torch.cat([vpn_e, asid_e], dim=-1)
            out = self.net.mlp(x)

            pfn_logits = out[:, :self.net.max_physical_frames]
            perm_logits = out[:, self.net.max_physical_frames:]

            loss_pfn = pfn_loss_fn(pfn_logits, target_pfns)
            loss_perm = perm_loss_fn(perm_logits, target_perms)
            loss = loss_pfn + loss_perm

            loss.backward()
            optimizer.step()

            # Check accuracy
            with torch.no_grad():
                pred_pfns = pfn_logits.argmax(dim=-1)
                acc = (pred_pfns == target_pfns).float().mean().item()
                if acc > best_acc:
                    best_acc = acc

        self.net.eval()
        self._trained = True

        stats = {
            "epochs": epochs,
            "num_mappings": len(valid_vpns),
            "final_accuracy": acc,
            "best_accuracy": best_acc,
            "final_loss": loss.item(),
        }
        logger.info(f"[MMU] Trained: {acc*100:.1f}% accuracy on {len(valid_vpns)} mappings")
        return stats

    # ─── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str = "models/research/neuros/mmu.pt"):
        """Save the trained neural page table."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        logger.info(f"[MMU] Saved to {path}")

    def load(self, path: str = "models/research/neuros/mmu.pt") -> bool:
        """Load a trained neural page table."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"[MMU] No model at {path}")
            return False
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.net.eval()
        self._trained = True
        logger.info(f"[MMU] Loaded from {path}")
        return True

    # ─── Diagnostics ──────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return MMU statistics."""
        return {
            "translations": self.translations,
            "page_faults": self.page_faults,
            "fault_rate": self.page_faults / max(1, self.translations),
            "mapped_pages": int((self.page_table_pfn[0] >= 0).sum().item()),
            "free_frames": self.free_frames,
            "trained": self._trained,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"NeuralMMU(pages={s['mapped_pages']}, "
                f"frames_free={s['free_frames']}, "
                f"faults={s['page_faults']}/{s['translations']}, "
                f"trained={s['trained']})")
