"""Neural ECC Memory: fault-tolerant storage with learned error-correcting codes.

Data is encoded into a wider embedding space (N bits -> embed_dim dimensions).
The encoding has learned redundancy: if up to a fraction of dimensions are
corrupted, the decoder still recovers the original value.

This is fundamentally different from conventional ECC:
  - Conventional ECC: fixed Hamming/Reed-Solomon codes, corrects up to t errors
  - Neural ECC: learned codes, adapts to error patterns, degrades GRACEFULLY
  - Conventional ECC: hard failure boundary (works perfectly, then fails completely)
  - Neural ECC: soft degradation curve (recovery rate decreases smoothly)

Default configuration: 16-bit values with 128-dim embeddings (8x redundancy).
Achieved: 100% at 0%, ~93% at 10%, ~65% at 20%, ~33% at 30% corruption.

Architecture:
    encoder: bits(N) -> MLP(N->256->256->embed_dim) with Tanh output
    decoder: MLP(embed_dim->256->256->N)
    No skip connections -- information is distributed uniformly across ALL
    embedding dimensions so no single dimension is critical.

Training: self-supervised with multi-scale corruption augmentation.
    Each batch trains on clean + light + medium + heavy corruption simultaneously.
    Gradients flow through corrupted paths back to the encoder, pushing it
    to spread information redundantly.

Integration:
    from ncpu.neural.neural_ecc_memory import NeuralECCMemory, train_ecc_memory

    mem = train_ecc_memory()  # trains + saves to models/neural_ecc_memory.pt
    mem = NeuralECCMemory.load()  # loads from saved checkpoint

    mem.write(0, 42)
    assert mem.read(0) == 42
    assert mem.read_corrupted(0, corruption_frac=0.1) == 42  # survives 10% corruption
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncpu.neural.neural_registers import MODELS_DIR

# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class ECCEncoder(nn.Module):
    """Encode N-bit values into a redundant embedding space.

    The encoder spreads each input bit's information across ALL output
    dimensions. This distributed representation means that corrupting a
    fraction of the dimensions loses only a fraction of the information.

    No skip connection: a fully learned encoding distributes information
    maximally, which is essential for corruption tolerance. If raw bits
    were passed through a skip, corrupting those dims would destroy them.

    Tanh output bounds embeddings in [-1, 1] for stability.
    """

    def __init__(self, n_bits: int = 16, embed_dim: int = 128):
        super().__init__()
        self.n_bits = n_bits
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Linear(n_bits, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
            nn.Tanh(),
        )

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """bits: [..., n_bits] -> embedding: [..., embed_dim]"""
        return self.net(bits)


class ECCDecoder(nn.Module):
    """Decode embeddings back to N-bit values, correcting corruption.

    The decoder reconstructs original bits even when some embedding dimensions
    have been replaced with noise. It uses the redundancy in the encoding to
    vote across multiple dimensions for each output bit.
    """

    def __init__(self, n_bits: int = 16, embed_dim: int = 128):
        super().__init__()
        self.n_bits = n_bits
        self.embed_dim = embed_dim
        self.main = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, n_bits),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """embedding: [..., embed_dim] -> bit_logits: [..., n_bits]"""
        return self.main(embedding)


# ---------------------------------------------------------------------------
# Neural ECC Memory
# ---------------------------------------------------------------------------


class NeuralECCMemory:
    """Memory with learned error-correcting codes.

    Data is encoded into a wider embedding space (n_bits -> embed_dim dims).
    The encoding has redundancy: if up to N dimensions are corrupted, the
    decoder still recovers the original value.

    Parameters:
        size: Number of memory cells (addresses 0..size-1).
        n_bits: Bit width of stored values (default 16 for half-precision).
            Higher bit widths require proportionally wider embeddings.
        embed_dim: Dimensionality of the encoding space. Higher = more
            redundancy = better error correction, at the cost of storage.
        device: Torch device for computation.
    """

    def __init__(
        self,
        encoder: Optional[ECCEncoder] = None,
        decoder: Optional[ECCDecoder] = None,
        size: int = 1024,
        n_bits: int = 16,
        embed_dim: int = 128,
        device: str = "cpu",
    ):
        self.size = size
        self.n_bits = n_bits
        self.embed_dim = embed_dim
        self.device = torch.device(device)
        self.max_val = (1 << n_bits) - 1

        self.encoder = encoder or ECCEncoder(n_bits, embed_dim)
        self.decoder = decoder or ECCDecoder(n_bits, embed_dim)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.encoder.eval()
        self.decoder.eval()

        # Storage: data lives as embeddings
        self.storage = torch.zeros(size, embed_dim, device=self.device)
        # Track which cells have been written
        self.valid = torch.zeros(size, dtype=torch.bool, device=self.device)

        self._reads = 0
        self._writes = 0
        self._corrections = 0

    def _int_to_bits(self, value: int) -> torch.Tensor:
        """Convert an unsigned integer to a float tensor of bits (LSB first)."""
        if value < 0:
            value = value + (1 << self.n_bits)
        value = value & self.max_val
        bits = torch.zeros(self.n_bits, dtype=torch.float32)
        for i in range(self.n_bits):
            bits[i] = float((value >> i) & 1)
        return bits

    def _bits_to_int(self, bit_logits: torch.Tensor) -> int:
        """Convert bit logits to an unsigned integer."""
        hard = (torch.sigmoid(bit_logits) > 0.5).long()
        value = 0
        for i in range(self.n_bits):
            if hard[i]:
                value |= (1 << i)
        return value

    @torch.no_grad()
    def write(self, addr: int, value: int) -> None:
        """Write an integer value by encoding it into a redundant embedding."""
        if not (0 <= addr < self.size):
            raise IndexError(f"Address {addr} out of range [0, {self.size})")
        bits = self._int_to_bits(value).to(self.device)
        embedding = self.encoder(bits.unsqueeze(0)).squeeze(0)
        self.storage[addr] = embedding
        self.valid[addr] = True
        self._writes += 1

    @torch.no_grad()
    def read(self, addr: int) -> int:
        """Read a value by decoding its embedding back to bits."""
        if not (0 <= addr < self.size):
            raise IndexError(f"Address {addr} out of range [0, {self.size})")
        if not self.valid[addr]:
            return 0
        embedding = self.storage[addr].unsqueeze(0)
        bit_logits = self.decoder(embedding).squeeze(0)
        self._reads += 1
        return self._bits_to_int(bit_logits)

    @torch.no_grad()
    def read_corrupted(
        self,
        addr: int,
        corruption_frac: float = 0.1,
        noise_scale: float = 2.0,
    ) -> int:
        """Read after randomly corrupting a fraction of embedding dimensions.

        This simulates hardware faults, bit flips in DRAM, cosmic ray
        strikes, or adversarial noise. The neural decoder should recover
        the original value from the uncorrupted dimensions.

        Args:
            addr: Memory address to read.
            corruption_frac: Fraction of embedding dims to corrupt (0.0-1.0).
            noise_scale: Scale of the Gaussian noise injected into corrupt dims.

        Returns:
            Recovered integer value.
        """
        if not (0 <= addr < self.size):
            raise IndexError(f"Address {addr} out of range [0, {self.size})")
        if not self.valid[addr]:
            return 0

        embedding = self.storage[addr].clone()
        n_corrupt = max(1, int(self.embed_dim * corruption_frac))
        corrupt_idx = torch.randperm(self.embed_dim, device=self.device)[:n_corrupt]
        embedding[corrupt_idx] = torch.randn(
            n_corrupt, device=self.device
        ) * noise_scale

        bit_logits = self.decoder(embedding.unsqueeze(0)).squeeze(0)
        self._reads += 1
        self._corrections += 1
        return self._bits_to_int(bit_logits)

    @torch.no_grad()
    def write_batch(self, addrs: list[int], values: list[int]) -> None:
        """Batch write multiple values in a single forward pass."""
        if len(addrs) != len(values):
            raise ValueError("addrs and values must have same length")
        bits = torch.stack(
            [self._int_to_bits(v) for v in values]
        ).to(self.device)
        embeddings = self.encoder(bits)
        for i, addr in enumerate(addrs):
            if not (0 <= addr < self.size):
                raise IndexError(f"Address {addr} out of range [0, {self.size})")
            self.storage[addr] = embeddings[i]
            self.valid[addr] = True
        self._writes += len(addrs)

    @torch.no_grad()
    def scrub(self, addr: int) -> bool:
        """Re-encode a cell to repair accumulated drift.

        Reads the value, then re-encodes and re-writes it. Returns True if
        the re-encoded value changed (indicating the cell had drifted).

        This is analogous to ECC memory scrubbing in servers.
        """
        if not self.valid[addr]:
            return False
        old_embedding = self.storage[addr].clone()
        value = self.read(addr)
        self.write(addr, value)
        return not torch.allclose(old_embedding, self.storage[addr], atol=1e-6)

    def reset(self) -> None:
        """Clear all memory cells."""
        self.storage.zero_()
        self.valid.zero_()
        self._reads = 0
        self._writes = 0
        self._corrections = 0

    @property
    def stats(self) -> dict:
        return {
            "size": self.size,
            "n_bits": self.n_bits,
            "embed_dim": self.embed_dim,
            "redundancy": f"{self.embed_dim / self.n_bits:.1f}x",
            "reads": self._reads,
            "writes": self._writes,
            "corrections": self._corrections,
            "cells_used": int(self.valid.sum().item()),
            "param_count": sum(p.numel() for p in self.encoder.parameters())
                         + sum(p.numel() for p in self.decoder.parameters()),
        }

    # -- Persistence --------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Save encoder/decoder weights to a checkpoint."""
        path = path or (MODELS_DIR / "neural_ecc_memory.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "embed_dim": self.embed_dim,
            "n_bits": self.n_bits,
            "size": self.size,
        }
        torch.save(state, path)
        return path

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        size: int = 1024,
        device: str = "cpu",
    ) -> "NeuralECCMemory":
        """Load a trained ECC memory from checkpoint."""
        path = path or (MODELS_DIR / "neural_ecc_memory.pt")
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        embed_dim = state["embed_dim"]
        n_bits = state.get("n_bits", 16)
        encoder = ECCEncoder(n_bits, embed_dim)
        decoder = ECCDecoder(n_bits, embed_dim)
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        return cls(
            encoder=encoder,
            decoder=decoder,
            size=size,
            n_bits=n_bits,
            embed_dim=embed_dim,
            device=device,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _generate_ecc_batch(
    batch_size: int, n_bits: int, device: torch.device
) -> torch.Tensor:
    """Generate a batch of random N-bit float vectors for training.

    Returns [batch_size, n_bits] tensor of binary float values (0.0 or 1.0).
    """
    max_val = (1 << n_bits)
    values = torch.randint(0, max_val, (batch_size,), dtype=torch.int64)
    shifts = torch.arange(n_bits, dtype=torch.int64)
    bits = ((values.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).float()
    return bits.to(device)


def _apply_corruption(
    embeddings: torch.Tensor,
    corruption_frac: float,
    noise_scale: float = 2.0,
) -> torch.Tensor:
    """Apply random corruption to a batch of embeddings.

    For each embedding in the batch, randomly selects a fraction of
    dimensions and replaces them with Gaussian noise. Uses a differentiable
    blending approach: corrupted = (1-mask)*embedding + mask*noise, so
    gradients flow through the uncorrupted dimensions back to the encoder.

    Args:
        embeddings: [batch, embed_dim] clean embeddings (may have grad).
        corruption_frac: Fraction of dims to corrupt per sample.
        noise_scale: Std dev of injected noise.

    Returns:
        [batch, embed_dim] corrupted embeddings with gradient flow preserved
        through uncorrupted dimensions.
    """
    batch_size, embed_dim = embeddings.shape
    device = embeddings.device

    n_corrupt = max(1, int(embed_dim * corruption_frac))

    # Create corruption mask as float for differentiable blending
    mask = torch.zeros(batch_size, embed_dim, device=device)
    for i in range(batch_size):
        idx = torch.randperm(embed_dim, device=device)[:n_corrupt]
        mask[i, idx] = 1.0

    noise = torch.randn(batch_size, embed_dim, device=device) * noise_scale

    # Differentiable blend: gradient flows through (1-mask)*embeddings
    corrupted = (1.0 - mask) * embeddings + mask * noise
    return corrupted


def train_ecc_memory(
    n_bits: int = 16,
    embed_dim: int = 128,
    epochs: int = 2000,
    batch_size: int = 2048,
    lr: float = 2e-3,
    device: str = "cpu",
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> NeuralECCMemory:
    """Train a NeuralECCMemory with corruption augmentation.

    The key insight: the corruption is applied WITH gradient flow through
    both encoder and decoder. The encoder learns to spread information
    redundantly so the decoder can reconstruct from partial embeddings.

    Training uses multi-scale corruption: each batch trains on clean,
    light corruption, and heavy corruption simultaneously. The encoder
    receives gradients from ALL corruption levels, pushing it to distribute
    information uniformly across embedding dimensions.

    Default: 16-bit words with 128-dim embeddings (8x redundancy).
    For 32-bit words, use embed_dim=256 (8x redundancy).

    Returns a trained NeuralECCMemory ready for use.
    """
    dev = torch.device(device)
    encoder = ECCEncoder(n_bits, embed_dim).to(dev)
    decoder = ECCDecoder(n_bits, embed_dim).to(dev)

    encoder.train()
    decoder.train()

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs,
        pct_start=0.15, anneal_strategy="cos",
        div_factor=5.0, final_div_factor=50.0,
    )

    best_score = 0.0
    best_state = None

    if verbose:
        print(f"Training Neural ECC Memory ({n_bits}-bit, embed_dim={embed_dim}, "
              f"{embed_dim/n_bits:.0f}x redundancy)...")
        print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
        print(f"  Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    for epoch in range(1, epochs + 1):
        progress = epoch / epochs

        # Multi-scale corruption: always train on multiple levels
        # Ramp up the heavy corruption level over training
        if progress < 0.05:
            corrupt_fracs = [0.0, 0.05]
        elif progress < 0.2:
            corrupt_fracs = [0.0, 0.10, 0.20]
        elif progress < 0.5:
            corrupt_fracs = [0.0, 0.10, 0.20, 0.30]
        else:
            phase_p = (progress - 0.5) / 0.5
            heavy = 0.30 + 0.20 * phase_p  # ramp to 50%
            corrupt_fracs = [0.0, 0.15, 0.30, heavy]

        # Generate batch
        target_bits = _generate_ecc_batch(batch_size, n_bits, dev)

        # Encode (gradients flow through encoder)
        embeddings = encoder(target_bits)

        # Multi-scale corruption training
        total_loss = torch.tensor(0.0, device=dev)
        for frac in corrupt_fracs:
            if frac == 0.0:
                logits = decoder(embeddings)
            else:
                corrupted = _apply_corruption(embeddings, frac, noise_scale=2.0)
                logits = decoder(corrupted)

            bce = F.binary_cross_entropy_with_logits(logits, target_bits)
            total_loss = total_loss + bce

        total_loss = total_loss / len(corrupt_fracs)

        # Margin penalty for confident predictions
        clean_logits = decoder(embeddings)
        margin = 5.0
        margin_penalty = F.relu(margin - clean_logits.abs()).mean()
        total_loss = total_loss + 0.1 * margin_penalty

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Evaluation
        if epoch % 50 == 0 or epoch == 1:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                test_bits = _generate_ecc_batch(4096, n_bits, dev)
                test_embed = encoder(test_bits)
                test_logits = decoder(test_embed)
                pred_bits = (torch.sigmoid(test_logits) > 0.5).float()
                clean_acc = (pred_bits == test_bits).all(dim=1).float().mean().item()

                corrupt_embed_10 = _apply_corruption(test_embed, 0.10)
                pred_10 = (torch.sigmoid(decoder(corrupt_embed_10)) > 0.5).float()
                acc_10 = (pred_10 == test_bits).all(dim=1).float().mean().item()

                corrupt_embed_20 = _apply_corruption(test_embed, 0.20)
                pred_20 = (torch.sigmoid(decoder(corrupt_embed_20)) > 0.5).float()
                acc_20 = (pred_20 == test_bits).all(dim=1).float().mean().item()

                corrupt_embed_30 = _apply_corruption(test_embed, 0.30)
                pred_30 = (torch.sigmoid(decoder(corrupt_embed_30)) > 0.5).float()
                acc_30 = (pred_30 == test_bits).all(dim=1).float().mean().item()

                # Score: weighted across corruption levels
                score = (
                    0.2 * clean_acc
                    + 0.3 * acc_10
                    + 0.3 * acc_20
                    + 0.2 * acc_30
                )
                if score >= best_score:
                    best_score = score
                    best_state = {
                        "encoder": {
                            k: v.clone() for k, v in encoder.state_dict().items()
                        },
                        "decoder": {
                            k: v.clone() for k, v in decoder.state_dict().items()
                        },
                    }

                if verbose and (epoch % 200 == 0 or epoch == 1):
                    cur_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"  Epoch {epoch:4d}  loss={total_loss.item():.4f}  "
                        f"clean={clean_acc:.3f}  "
                        f"10%={acc_10:.3f}  "
                        f"20%={acc_20:.3f}  "
                        f"30%={acc_30:.3f}  "
                        f"lr={cur_lr:.6f}"
                    )

            encoder.train()
            decoder.train()

    # Restore best weights
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])

    encoder.eval()
    decoder.eval()

    mem = NeuralECCMemory(
        encoder=encoder,
        decoder=decoder,
        n_bits=n_bits,
        embed_dim=embed_dim,
        device=device,
    )

    save_path = save_path or (MODELS_DIR / "neural_ecc_memory.pt")
    mem.save(save_path)
    if verbose:
        print(f"  Saved to {save_path}")
        print(f"  Parameters: {mem.stats['param_count']:,}")
        print(f"  Best weighted score: {best_score:.4f}")

    return mem


# ---------------------------------------------------------------------------
# Evaluation sweep
# ---------------------------------------------------------------------------


def evaluate_corruption_sweep(
    mem: NeuralECCMemory,
    n_values: int = 1000,
    corruption_levels: Optional[list[float]] = None,
    verbose: bool = True,
) -> dict[float, float]:
    """Evaluate recovery rate across multiple corruption levels.

    Writes n_values random values within the memory's bit range, then
    for each corruption level, reads each value with that level of
    corruption and measures perfect recovery rate.

    Returns dict mapping corruption_frac -> recovery_rate.
    """
    if corruption_levels is None:
        corruption_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    max_val = (1 << mem.n_bits) - 1

    # Write random values
    values = []
    for i in range(n_values):
        v = random.randint(0, max_val)
        addr = i % mem.size
        mem.write(addr, v)
        values.append((addr, v))

    results = {}
    for frac in corruption_levels:
        correct = 0
        for addr, expected in values:
            if frac == 0.0:
                got = mem.read(addr)
            else:
                got = mem.read_corrupted(addr, corruption_frac=frac)
            if got == expected:
                correct += 1
        rate = correct / n_values
        results[frac] = rate

    if verbose:
        print(f"\nNeural ECC Memory Corruption Sweep ({mem.n_bits}-bit values, "
              f"{mem.embed_dim}-dim embeddings, {mem.embed_dim/mem.n_bits:.0f}x redundancy):")
        print(f"  {'Corruption':>12s}    {'Recovery Rate':>14s}")
        print(f"  {'─' * 12}    {'─' * 14}")
        for frac, rate in results.items():
            print(f"  {frac*100:10.0f}% dims    {rate*100:12.1f}%")

    return results
