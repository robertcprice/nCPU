"""Content-Addressable Neural Memory: query by VALUE, not just by address.

Standard RAM answers: "What value is at address X?"
Neural CAM also answers: "What address holds a value closest to X?"

This is a neural embedding-based content-addressable memory that supports:
  1. Address query: read(addr) -> value (like normal RAM)
  2. Content query: search(value) -> [(addr, stored_value, similarity)]
  3. Duplicate detection: find_duplicates(threshold) -> pairs
  4. Range query: search_range(lo, hi) -> addresses with values in range

The value encoder is trained with contrastive learning so that numerically
close values have similar embeddings and distant values have dissimilar
embeddings. This enables approximate nearest-neighbor search in value space.

Use cases in a neural CPU:
  - Register renaming: find a register already holding a value we need
  - Value prediction: which memory cell likely holds this constant?
  - Deduplication: detect when two cells store equivalent data
  - Cache optimization: content-based cache lookup (like TLB but for values)

Architecture:
    value_encoder: bits(64) -> MLP(64->128->64) with normalized output
    Training: contrastive loss with curriculum difficulty

Integration:
    from ncpu.neural.neural_cam import NeuralCAM, train_cam

    cam = train_cam()  # trains + saves to models/neural_cam.pt
    cam = NeuralCAM.load()

    cam.write(0, 42)
    cam.write(1, 43)
    cam.write(2, 1000)
    results = cam.search(42)  # -> [(0, 42, 1.00), (1, 43, 0.99), ...]
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncpu.neural.neural_registers import (
    int_to_bits,
    bits_to_int,
    int_to_bits_batch,
    MODELS_DIR,
)

# ---------------------------------------------------------------------------
# Value Encoder
# ---------------------------------------------------------------------------


class ValueEncoder(nn.Module):
    """Encode int64 values into a normalized embedding space.

    The encoder maps 64-bit values to unit-norm vectors such that numerically
    close values have similar embeddings (high cosine similarity) and distant
    values have dissimilar embeddings.

    L2-normalization of the output ensures cosine similarity equals dot
    product, simplifying search operations.

    Architecture: 64 -> 128 -> 128 -> embed_dim, with LayerNorm + GELU.
    Output is L2-normalized.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """bits: [..., 64] -> embedding: [..., embed_dim] (L2-normalized)."""
        raw = self.net(bits)
        return F.normalize(raw, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Content-Addressable Memory
# ---------------------------------------------------------------------------


class NeuralCAM:
    """Content-Addressable Memory using neural embeddings.

    Stores key-value pairs where keys are addresses and values are data.
    Supports both address-based and content-based queries.

    Parameters:
        size: Number of memory cells.
        embed_dim: Dimensionality of value embeddings.
        device: Torch device.
    """

    def __init__(
        self,
        encoder: Optional[ValueEncoder] = None,
        size: int = 256,
        embed_dim: int = 64,
        device: str = "cpu",
    ):
        self.size = size
        self.embed_dim = embed_dim
        self.device = torch.device(device)

        self.encoder = encoder or ValueEncoder(embed_dim)
        self.encoder.to(self.device)
        self.encoder.eval()

        # Storage
        self.embeddings = torch.zeros(
            size, embed_dim, device=self.device
        )
        self.raw_values = torch.zeros(size, dtype=torch.int64, device=self.device)
        self.valid = torch.zeros(size, dtype=torch.bool, device=self.device)

        self._reads = 0
        self._writes = 0
        self._searches = 0

    @torch.no_grad()
    def write(self, addr: int, value: int) -> None:
        """Write a value to an address, computing its embedding."""
        if not (0 <= addr < self.size):
            raise IndexError(f"Address {addr} out of range [0, {self.size})")
        bits = int_to_bits(value, 64).to(self.device)
        self.embeddings[addr] = self.encoder(bits.unsqueeze(0)).squeeze(0)
        self.raw_values[addr] = value
        self.valid[addr] = True
        self._writes += 1

    def read(self, addr: int) -> int:
        """Read a value by address (standard RAM operation)."""
        if not (0 <= addr < self.size):
            raise IndexError(f"Address {addr} out of range [0, {self.size})")
        if not self.valid[addr]:
            return 0
        self._reads += 1
        return int(self.raw_values[addr].item())

    @torch.no_grad()
    def search(
        self,
        target_value: int,
        top_k: int = 3,
    ) -> list[tuple[int, int, float]]:
        """Find addresses storing values closest to target_value.

        Uses cosine similarity in embedding space for approximate nearest
        neighbor search. Because the encoder was trained with contrastive
        learning, similar values have similar embeddings.

        Args:
            target_value: The value to search for.
            top_k: Number of results to return.

        Returns:
            List of (addr, stored_value, similarity) tuples, sorted by
            descending similarity.
        """
        if not self.valid.any():
            return []

        target_bits = int_to_bits(target_value, 64).to(self.device)
        target_embed = self.encoder(target_bits.unsqueeze(0)).squeeze(0)

        # Cosine similarity against all valid entries
        valid_mask = self.valid
        valid_indices = torch.where(valid_mask)[0]
        valid_embeddings = self.embeddings[valid_mask]

        sims = F.cosine_similarity(
            target_embed.unsqueeze(0),
            valid_embeddings,
            dim=1,
        )

        k = min(top_k, len(sims))
        top_sims, top_idx = sims.topk(k)

        results = []
        for sim_val, idx in zip(top_sims, top_idx):
            addr = int(valid_indices[idx].item())
            val = int(self.raw_values[addr].item())
            results.append((addr, val, float(sim_val.item())))

        self._searches += 1
        return results

    @torch.no_grad()
    def find_duplicates(
        self,
        threshold: float = 0.99,
    ) -> list[tuple[int, int, float]]:
        """Find pairs of addresses storing identical or near-identical values.

        Computes the pairwise cosine similarity matrix over all valid entries
        and returns pairs above the threshold.

        Args:
            threshold: Minimum cosine similarity to consider a duplicate.

        Returns:
            List of (addr_a, addr_b, similarity) tuples.
        """
        if self.valid.sum() < 2:
            return []

        valid_indices = torch.where(self.valid)[0]
        valid_embeddings = self.embeddings[self.valid]

        # Pairwise cosine similarity
        sim_matrix = F.cosine_similarity(
            valid_embeddings.unsqueeze(1),
            valid_embeddings.unsqueeze(0),
            dim=2,
        )

        # Extract upper triangle (avoid self-pairs and duplicates)
        n = len(valid_indices)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                s = float(sim_matrix[i, j].item())
                if s >= threshold:
                    addr_a = int(valid_indices[i].item())
                    addr_b = int(valid_indices[j].item())
                    pairs.append((addr_a, addr_b, s))

        return sorted(pairs, key=lambda x: -x[2])

    @torch.no_grad()
    def search_range(
        self,
        lo: int,
        hi: int,
        max_results: int = 10,
    ) -> list[tuple[int, int, float]]:
        """Find addresses storing values in [lo, hi] (approximate).

        Encodes the midpoint of the range and searches by similarity.
        Results may include values slightly outside the range due to
        the approximate nature of the neural embedding.

        Args:
            lo: Lower bound of the search range (inclusive).
            hi: Upper bound of the search range (inclusive).
            max_results: Maximum number of results.

        Returns:
            List of (addr, stored_value, similarity) tuples.
        """
        midpoint = (lo + hi) // 2
        candidates = self.search(midpoint, top_k=max_results * 2)
        # Filter to values actually in range
        in_range = [
            (addr, val, sim)
            for addr, val, sim in candidates
            if lo <= val <= hi
        ]
        return in_range[:max_results]

    @torch.no_grad()
    def write_batch(self, addrs: list[int], values: list[int]) -> None:
        """Batch write multiple values in a single forward pass."""
        if len(addrs) != len(values):
            raise ValueError("addrs and values must have same length")
        bits = torch.stack([int_to_bits(v, 64) for v in values]).to(self.device)
        embeddings = self.encoder(bits)
        for i, (addr, val) in enumerate(zip(addrs, values)):
            if not (0 <= addr < self.size):
                raise IndexError(f"Address {addr} out of range [0, {self.size})")
            self.embeddings[addr] = embeddings[i]
            self.raw_values[addr] = val
            self.valid[addr] = True
        self._writes += len(addrs)

    def clear(self, addr: int) -> None:
        """Invalidate a single cell."""
        if not (0 <= addr < self.size):
            raise IndexError(f"Address {addr} out of range [0, {self.size})")
        self.valid[addr] = False

    def reset(self) -> None:
        """Clear all cells."""
        self.embeddings.zero_()
        self.raw_values.zero_()
        self.valid.zero_()
        self._reads = 0
        self._writes = 0
        self._searches = 0

    @property
    def stats(self) -> dict:
        return {
            "size": self.size,
            "embed_dim": self.embed_dim,
            "reads": self._reads,
            "writes": self._writes,
            "searches": self._searches,
            "cells_used": int(self.valid.sum().item()),
            "param_count": sum(p.numel() for p in self.encoder.parameters()),
        }

    # -- Persistence --------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Save encoder weights to a checkpoint."""
        path = path or (MODELS_DIR / "neural_cam.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "encoder": self.encoder.state_dict(),
            "embed_dim": self.embed_dim,
        }
        torch.save(state, path)
        return path

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        size: int = 256,
        device: str = "cpu",
    ) -> "NeuralCAM":
        """Load a trained CAM from checkpoint."""
        path = path or (MODELS_DIR / "neural_cam.pt")
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        embed_dim = state["embed_dim"]
        encoder = ValueEncoder(embed_dim)
        encoder.load_state_dict(state["encoder"])
        return cls(encoder=encoder, size=size, embed_dim=embed_dim, device=device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _value_to_float(value: int) -> float:
    """Map int64 to a normalized float for distance computation.

    Uses log-scale mapping to handle the enormous int64 range while
    preserving local ordering: nearby integers map to nearby floats.
    """
    if value == 0:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * torch.log1p(torch.tensor(abs(value), dtype=torch.float64)).item()


def train_cam(
    embed_dim: int = 64,
    epochs: int = 1500,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cpu",
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> NeuralCAM:
    """Train the CAM value encoder with contrastive learning.

    The encoder learns to map numerically close values to similar
    embeddings and distant values to dissimilar embeddings.

    Training uses a margin-based contrastive loss:
    - Positive pairs: (x, x+delta) where delta is small -> pull together
    - Negative pairs: (x, y) where |x-y| >> delta -> push apart

    Curriculum: starts with large value ranges (easy separation) and
    gradually introduces finer distinctions.

    Returns a trained NeuralCAM ready for use.
    """
    dev = torch.device(device)
    encoder = ValueEncoder(embed_dim).to(dev)
    encoder.train()

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    if verbose:
        print(f"Training Neural CAM (embed_dim={embed_dim})...")
        print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    for epoch in range(1, epochs + 1):
        # Generate anchor values
        anchors = torch.randint(
            -(1 << 30), (1 << 30), (batch_size,), dtype=torch.int64
        )

        # Positive pairs: anchor + small delta
        # Curriculum: start with delta up to 100, narrow to delta up to 5
        progress = epoch / epochs
        max_delta = max(2, int(100 * (1 - progress) + 5 * progress))
        deltas = torch.randint(1, max_delta + 1, (batch_size,), dtype=torch.int64)
        signs = torch.randint(0, 2, (batch_size,), dtype=torch.int64) * 2 - 1
        positives = anchors + deltas * signs

        # Negative pairs: random values far away
        negatives = torch.randint(
            -(1 << 30), (1 << 30), (batch_size,), dtype=torch.int64
        )
        # Ensure negatives are actually far from anchors
        too_close = (negatives - anchors).abs() < 1000
        negatives[too_close] = anchors[too_close] + torch.randint(
            10000, 100000, (too_close.sum(),), dtype=torch.int64
        )

        # Convert to bits
        anchor_bits = int_to_bits_batch(anchors.to(dev), 64)
        pos_bits = int_to_bits_batch(positives.to(dev), 64)
        neg_bits = int_to_bits_batch(negatives.to(dev), 64)

        # Encode
        anchor_emb = encoder(anchor_bits)
        pos_emb = encoder(pos_bits)
        neg_emb = encoder(neg_bits)

        # Contrastive loss: triplet margin
        # Pull anchor-positive together, push anchor-negative apart
        pos_dist = 1 - F.cosine_similarity(anchor_emb, pos_emb, dim=1)
        neg_dist = 1 - F.cosine_similarity(anchor_emb, neg_emb, dim=1)

        # Triplet loss with margin
        margin = 0.5
        triplet_loss = F.relu(pos_dist - neg_dist + margin).mean()

        # Additional: pull identical values together (self-consistency)
        anchor_emb2 = encoder(anchor_bits)  # re-encode same values
        consistency_loss = (1 - F.cosine_similarity(anchor_emb, anchor_emb2, dim=1)).mean()

        # Proportional similarity: closer values should have more similar embeddings
        # Use the actual deltas to weight the positive distance
        delta_magnitude = deltas.float().to(dev)
        # Normalize deltas to [0, 1] range
        delta_norm = delta_magnitude / (max_delta + 1)
        # Weighted positive distance: small deltas should have small distances
        proportional_loss = ((1 - delta_norm) * pos_dist).mean()

        loss = triplet_loss + 0.1 * consistency_loss + 0.3 * proportional_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if verbose and (epoch % 300 == 0 or epoch == 1):
            with torch.no_grad():
                avg_pos_sim = (1 - pos_dist).mean().item()
                avg_neg_sim = (1 - neg_dist).mean().item()
                print(
                    f"  Epoch {epoch:4d}  loss={loss.item():.4f}  "
                    f"pos_sim={avg_pos_sim:.3f}  neg_sim={avg_neg_sim:.3f}  "
                    f"delta_range=[1,{max_delta}]"
                )

    encoder.eval()
    cam = NeuralCAM(encoder=encoder, embed_dim=embed_dim, device=device)

    save_path = save_path or (MODELS_DIR / "neural_cam.pt")
    cam.save(save_path)
    if verbose:
        print(f"  Saved to {save_path}")
        print(f"  Parameters: {cam.stats['param_count']:,}")

    return cam


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_cam(
    cam: NeuralCAM,
    verbose: bool = True,
) -> dict:
    """Evaluate the CAM on search accuracy and duplicate detection.

    Tests:
    1. Exact match: can we find the address of a value we just wrote?
    2. Approximate match: does searching for x find x+1 as second result?
    3. Ordering: are results ordered by value closeness?
    4. Duplicate detection: can we find cells with identical values?

    Returns dict with evaluation metrics.
    """
    cam.reset()

    # Write a set of test values
    test_values = list(range(0, 100, 2))  # even numbers 0..98
    for i, v in enumerate(test_values):
        cam.write(i, v)

    # Also write some duplicates
    cam.write(60, 42)
    cam.write(61, 42)
    cam.write(62, 84)

    metrics = {}

    # Test 1: Exact match
    exact_hits = 0
    for i, v in enumerate(test_values):
        results = cam.search(v, top_k=1)
        if results and results[0][1] == v:
            exact_hits += 1
    metrics["exact_match_rate"] = exact_hits / len(test_values)

    # Test 2: Approximate match (search for odd number, find neighbors)
    approx_hits = 0
    for v in range(1, 99, 2):  # odd numbers
        results = cam.search(v, top_k=2)
        if results:
            found_vals = {r[1] for r in results}
            # Should find v-1 or v+1 (the nearest even numbers)
            if (v - 1) in found_vals or (v + 1) in found_vals:
                approx_hits += 1
    metrics["approx_match_rate"] = approx_hits / 49

    # Test 3: Ordering check (top result should be closest value)
    ordering_correct = 0
    n_ordering_tests = 20
    for _ in range(n_ordering_tests):
        query = random.randint(0, 98)
        results = cam.search(query, top_k=3)
        if len(results) >= 2:
            dist_1 = abs(results[0][1] - query)
            dist_2 = abs(results[1][1] - query)
            if dist_1 <= dist_2:
                ordering_correct += 1
    metrics["ordering_accuracy"] = ordering_correct / n_ordering_tests

    # Test 4: Duplicate detection
    duplicates = cam.find_duplicates(threshold=0.99)
    dup_pairs = {(min(a, b), max(a, b)) for a, b, s in duplicates}
    expected_dup = {(21, 60), (21, 61), (60, 61)}  # addr 21 has val 42 too
    metrics["duplicates_found"] = len(duplicates)

    if verbose:
        print("\nNeural CAM Evaluation:")
        print(f"  Exact match rate:    {metrics['exact_match_rate']:.1%}")
        print(f"  Approx match rate:   {metrics['approx_match_rate']:.1%}")
        print(f"  Ordering accuracy:   {metrics['ordering_accuracy']:.1%}")
        print(f"  Duplicates found:    {metrics['duplicates_found']}")

        # Show a sample search
        print("\n  Sample searches:")
        for query_val in [42, 43, 0, 99]:
            results = cam.search(query_val, top_k=3)
            result_str = ", ".join(
                f"addr {a} (val={v}, sim={s:.2f})" for a, v, s in results
            )
            print(f"    search({query_val}) -> {result_str}")

    return metrics
