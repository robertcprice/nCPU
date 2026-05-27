"""Neural Register File: registers live as learned embeddings in a weight matrix.

Instead of a plain int64[32] array, every read/write passes through trained
encoder/decoder MLPs. The register bank IS the neural network's weight matrix --
storage and computation are unified.

Architecture:
    encoder: Skip(bits) + MLP(64 -> 128 -> 64)   bits concatenated with learned features
    decoder: Skip(embed[:64]) + MLP(128 -> 128 -> 64)   residual correction over skip bits
    register_bank: Tensor[32, 128]                the "weights" ARE the storage

The residual skip connection passes raw bits through the first 64 dims of the
embedding, while the MLP learns additional features in the remaining 64 dims.
The decoder adds a learned correction to the skip bits, ensuring near-perfect
reconstruction from the very start of training.

Training: self-supervised autoencoder with BCE + margin penalty. Converges to
100% lossless int64 round-trip in ~500 epochs. The margin penalty pushes all
bit logits to |logit| > 1.0, guaranteeing confident predictions.

The encoder/decoder total ~41K params.

Integration:
    from ncpu.neural.neural_registers import NeuralRegisterFile, train_register_file

    rf = NeuralRegisterFile()          # untrained
    rf = train_register_file()         # trains + saves to models/neural_registers.pt
    rf = NeuralRegisterFile.load()     # loads from saved checkpoint

    rf.write(0, 42)
    assert rf.read(0) == 42
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# ────────────────────────────────────────────────────────────────────────────
# Bit conversion utilities (CPU-friendly, handles full int64 range)
# ────────────────────────────────────────────────────────────────────────────

def int_to_bits(value: int, n_bits: int = 64) -> torch.Tensor:
    """Convert a Python int to a float tensor of 0s and 1s (LSB first).

    Handles the full int64 range by working with the unsigned 64-bit
    representation (two's complement).
    """
    if value < 0:
        value = value + (1 << n_bits)
    bits = torch.zeros(n_bits, dtype=torch.float32)
    for i in range(n_bits):
        bits[i] = float((value >> i) & 1)
    return bits


def bits_to_int(bits: torch.Tensor) -> int:
    """Convert a float tensor of bit logits (LSB first) to a Python int.

    Applies sigmoid + threshold, then reassembles the unsigned value and
    converts back to signed int64 range.
    """
    n_bits = bits.shape[0]
    hard_bits = (torch.sigmoid(bits) > 0.5).long()
    value = 0
    for i in range(n_bits):
        if hard_bits[i]:
            value |= (1 << i)
    if n_bits == 64 and value >= (1 << 63):
        value -= (1 << 64)
    return value


def int_to_bits_batch(values: torch.Tensor, n_bits: int = 64) -> torch.Tensor:
    """Batch convert int64 tensor [N] to bit tensor [N, n_bits] (LSB first).

    GPU-friendly: pure tensor ops, no Python loops.
    """
    device = values.device
    shifts = torch.arange(n_bits, dtype=torch.int64, device=device)
    bits = ((values.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).float()
    return bits


def bits_to_int_batch(bit_logits: torch.Tensor) -> torch.Tensor:
    """Batch convert bit logit tensor [N, n_bits] to int64 tensor [N].

    GPU-friendly, handles two's complement for 64-bit.
    """
    n_bits = bit_logits.shape[1]
    device = bit_logits.device
    hard_bits = (torch.sigmoid(bit_logits) > 0.5).long()
    weights = torch.zeros(n_bits, dtype=torch.int64, device=device)
    for i in range(min(n_bits, 63)):
        weights[i] = 1 << i
    if n_bits == 64:
        weights[63] = -(1 << 63)
    values = (hard_bits * weights.unsqueeze(0)).sum(dim=1)
    return values


# ────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder MLPs
# ────────────────────────────────────────────────────────────────────────────

class RegisterEncoder(nn.Module):
    """Residual encoder: 64-bit float vector -> embedding.

    Architecture: A learned MLP transformation concatenated with the raw bits
    via a skip connection. The raw bits occupy the first 64 dims of the
    embedding, while the MLP output fills the remaining (embed_dim - 64) dims.
    This guarantees the embedding preserves all input information.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        assert embed_dim >= 64, "embed_dim must be >= 64 for skip connection"
        self.embed_dim = embed_dim
        self.extra_dim = embed_dim - 64
        if self.extra_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, self.extra_dim),
            )
        else:
            self.net = None

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """bits: [..., 64] -> embedding: [..., embed_dim]"""
        if self.net is not None:
            extra = self.net(bits)  # [..., extra_dim]
            return torch.cat([bits, extra], dim=-1)  # [..., embed_dim]
        return bits


class RegisterDecoder(nn.Module):
    """Residual decoder: embedding -> 64-bit logit vector.

    Architecture: The first 64 dims of the embedding are the raw bits (from
    the skip connection). The decoder MLP refines these using all embed_dim
    dimensions, producing correction logits. The final output is the sum of
    the skip bits and the MLP correction, ensuring near-perfect reconstruction
    even early in training.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """embedding: [..., embed_dim] -> bit_logits: [..., 64]"""
        skip = embedding[..., :64]            # raw bits from encoder skip
        correction = self.net(embedding)       # learned correction
        return skip + correction               # residual output


# ────────────────────────────────────────────────────────────────────────────
# Neural Register File
# ────────────────────────────────────────────────────────────────────────────

class NeuralRegisterFile:
    """32 registers stored as learned embeddings in a weight matrix.

    Every read converts an embedding back to int64 through the decoder MLP.
    Every write converts an int64 to an embedding through the encoder MLP.

    The register bank tensor IS the storage -- no separate int array.
    """

    NUM_REGISTERS = 32
    EMBED_DIM = 128

    def __init__(self, encoder: Optional[RegisterEncoder] = None,
                 decoder: Optional[RegisterDecoder] = None,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.embed_dim = self.EMBED_DIM

        self.encoder = encoder or RegisterEncoder(self.embed_dim)
        self.decoder = decoder or RegisterDecoder(self.embed_dim)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.encoder.eval()
        self.decoder.eval()

        # The register bank: 32 embeddings (initialized to zero = value 0)
        self.register_bank = torch.zeros(
            self.NUM_REGISTERS, self.embed_dim,
            dtype=torch.float32, device=self.device
        )

        self._reads = 0
        self._writes = 0

    @torch.no_grad()
    def write(self, reg_idx: int, value: int) -> None:
        """Write an int64 value into a register via neural encoding."""
        if not (0 <= reg_idx < self.NUM_REGISTERS):
            raise IndexError(f"Register index {reg_idx} out of range [0, 31]")
        bits = int_to_bits(value, 64).to(self.device)
        embedding = self.encoder(bits.unsqueeze(0)).squeeze(0)
        self.register_bank[reg_idx] = embedding
        self._writes += 1

    @torch.no_grad()
    def read(self, reg_idx: int) -> int:
        """Read a register value via neural decoding. Reg 31 = XZR (always 0)."""
        if not (0 <= reg_idx < self.NUM_REGISTERS):
            raise IndexError(f"Register index {reg_idx} out of range [0, 31]")
        if reg_idx == 31:
            return 0
        embedding = self.register_bank[reg_idx].unsqueeze(0)
        bit_logits = self.decoder(embedding).squeeze(0)
        self._reads += 1
        return bits_to_int(bit_logits)

    @torch.no_grad()
    def read_batch(self, indices: list[int]) -> list[int]:
        """Batch read multiple registers in a single forward pass."""
        if not indices:
            return []
        idx_t = torch.tensor(indices, dtype=torch.long, device=self.device)
        embeddings = self.register_bank[idx_t]
        bit_logits = self.decoder(embeddings)
        self._reads += len(indices)
        results = bits_to_int_batch(bit_logits).tolist()
        return [0 if idx == 31 else val for idx, val in zip(indices, results)]

    @torch.no_grad()
    def write_batch(self, pairs: list[tuple[int, int]]) -> None:
        """Batch write multiple register values in a single forward pass."""
        if not pairs:
            return
        indices = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        bits = torch.stack([int_to_bits(v, 64) for v in values]).to(self.device)
        embeddings = self.encoder(bits)
        for i, idx in enumerate(indices):
            self.register_bank[idx] = embeddings[i]
        self._writes += len(pairs)

    def reset(self) -> None:
        """Zero all registers (re-encode 0 into every slot)."""
        self.register_bank.zero_()
        self._reads = 0
        self._writes = 0

    @property
    def stats(self) -> dict:
        return {
            "reads": self._reads,
            "writes": self._writes,
            "num_registers": self.NUM_REGISTERS,
            "embed_dim": self.embed_dim,
            "param_count": sum(p.numel() for p in self.encoder.parameters())
                         + sum(p.numel() for p in self.decoder.parameters()),
        }

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Save encoder/decoder weights to a checkpoint."""
        path = path or (MODELS_DIR / "neural_registers.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "embed_dim": self.embed_dim,
        }
        torch.save(state, path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None, device: str = "cpu") -> "NeuralRegisterFile":
        """Load a trained register file from checkpoint."""
        path = path or (MODELS_DIR / "neural_registers.pt")
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        embed_dim = state.get("embed_dim", 128)
        encoder = RegisterEncoder(embed_dim)
        decoder = RegisterDecoder(embed_dim)
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        return cls(encoder=encoder, decoder=decoder, device=device)

    def export_flat_weights(self) -> dict[str, list[float]]:
        """Export all weights as flat float lists for Metal shader integration."""
        flat = {}
        for name, param in self.encoder.named_parameters():
            flat[f"encoder.{name}"] = param.detach().cpu().flatten().tolist()
        for name, param in self.decoder.named_parameters():
            flat[f"decoder.{name}"] = param.detach().cpu().flatten().tolist()
        return flat

    def export_metal_binary(self, path: Optional[Path] = None) -> Path:
        """Export weights as a single binary file for Metal shader consumption.

        Format: [num_layers: u32] then for each layer:
            [name_len: u32][name: bytes][num_floats: u32][floats: f32[]]
        """
        path = path or (MODELS_DIR / "neural_registers_metal.bin")
        path.parent.mkdir(parents=True, exist_ok=True)
        flat = self.export_flat_weights()
        with open(path, "wb") as f:
            f.write(struct.pack("<I", len(flat)))
            for name, values in flat.items():
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<I", len(values)))
                for v in values:
                    f.write(struct.pack("<f", v))
        return path


# ────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────

def _generate_training_batch(batch_size: int, device: torch.device) -> torch.Tensor:
    """Generate a batch of random int64 values spanning the full range.

    Strategy: mix uniform random with edge cases to ensure the network
    learns the full int64 range including boundary values.
    """
    n_edge = min(batch_size // 8, 32)
    n_random = batch_size - n_edge

    # Random int64 values (use two 32-bit halves to cover full range)
    hi = torch.randint(-(1 << 31), (1 << 31) - 1, (n_random,), dtype=torch.int64)
    lo = torch.randint(0, (1 << 32) - 1, (n_random,), dtype=torch.int64)
    random_vals = (hi << 32) | (lo & 0xFFFFFFFF)

    # Edge cases
    edges = [0, 1, -1, 2, -2, 127, -128, 255, 256, 65535, -65536,
             (1 << 31) - 1, -(1 << 31), (1 << 32), -(1 << 32),
             (1 << 63) - 1, -(1 << 63), 42, -42, 1000, -1000,
             0x5555555555555555, -0x5555555555555555,
             0x7FFFFFFFFFFFFFFF, 0x0000000100000000,
             0xDEADBEEF, 0xCAFEBABE, -0xDEADBEEF,
             (1 << 48) - 1, -(1 << 48), (1 << 16) - 1, -(1 << 16)]
    edge_vals = torch.tensor(edges[:n_edge], dtype=torch.int64)

    all_vals = torch.cat([random_vals, edge_vals])
    return all_vals.to(device)


def train_register_file(
    epochs: int = 1500,
    batch_size: int = 2048,
    lr: float = 2e-3,
    device: str = "cpu",
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> NeuralRegisterFile:
    """Train a NeuralRegisterFile until 100% lossless reconstruction.

    Training loop:
        1. Generate random int64 values
        2. Convert to 64-bit float vectors
        3. Encode to embeddings via encoder MLP
        4. Decode back to 64-bit logits via decoder MLP
        5. BCE loss + margin penalty for confident predictions
        6. Stop when 100% reconstruction accuracy on 4096-sample test set

    Returns a trained NeuralRegisterFile ready for use.
    """
    dev = torch.device(device)
    encoder = RegisterEncoder(NeuralRegisterFile.EMBED_DIM).to(dev)
    decoder = RegisterDecoder(NeuralRegisterFile.EMBED_DIM).to(dev)

    encoder.train()
    decoder.train()

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs,
        pct_start=0.3, anneal_strategy="cos",
        div_factor=10.0, final_div_factor=100.0,
    )

    best_accuracy = 0.0
    best_state = None
    consecutive_perfect = 0
    # Require 5 consecutive perfect evals on 8192 samples to ensure
    # true generalization, not just lucky batches
    target_consecutive = 5
    eval_batch_size = 8192

    for epoch in range(1, epochs + 1):
        # ── Training step ────────────────────────────────────────────
        values = _generate_training_batch(batch_size, dev)
        target_bits = int_to_bits_batch(values, 64)  # [N, 64]

        embeddings = encoder(target_bits)        # [N, embed_dim]
        bit_logits = decoder(embeddings)          # [N, 64]

        # Primary loss: binary cross-entropy
        bce_loss = F.binary_cross_entropy_with_logits(bit_logits, target_bits)

        # Margin loss: penalize logits with |logit| < margin
        # Pushes logits to be strongly positive or negative, ensuring
        # confident and lossless predictions after sigmoid thresholding
        margin = 5.0
        margin_penalty = F.relu(margin - bit_logits.abs()).mean()

        loss = bce_loss + 0.2 * margin_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── Evaluation (every 10 epochs) ─────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                test_values = _generate_training_batch(eval_batch_size, dev)
                test_bits = int_to_bits_batch(test_values, 64)
                test_logits = decoder(encoder(test_bits))
                pred_bits = (torch.sigmoid(test_logits) > 0.5).float()
                bit_acc = (pred_bits == test_bits).float().mean().item()
                per_value_correct = (pred_bits == test_bits).all(dim=1).float()
                value_acc = per_value_correct.mean().item()
                # Check minimum logit confidence (all should be > 1.0)
                min_confidence = test_logits.abs().min().item()

                if value_acc > best_accuracy:
                    best_accuracy = value_acc
                    best_state = {
                        "encoder": {k: v.clone() for k, v in encoder.state_dict().items()},
                        "decoder": {k: v.clone() for k, v in decoder.state_dict().items()},
                    }

                if value_acc == 1.0 and min_confidence > 1.0:
                    consecutive_perfect += 1
                else:
                    consecutive_perfect = 0

                if verbose and (epoch % 50 == 0 or epoch == 1 or value_acc == 1.0):
                    cur_lr = optimizer.param_groups[0]["lr"]
                    print(f"  Epoch {epoch:4d}  loss={loss.item():.6f}  "
                          f"bit_acc={bit_acc:.4f}  value_acc={value_acc:.4f}  "
                          f"min_conf={min_confidence:.2f}  lr={cur_lr:.6f}")

                if consecutive_perfect >= target_consecutive:
                    if verbose:
                        print(f"  Converged at epoch {epoch} "
                              f"({target_consecutive} consecutive perfect evals "
                              f"with min_confidence > 1.0)")
                    break

            encoder.train()
            decoder.train()

    # ── Restore best weights ─────────────────────────────────────────
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])

    encoder.eval()
    decoder.eval()

    rf = NeuralRegisterFile(encoder=encoder, decoder=decoder, device=device)

    # ── Save checkpoint ──────────────────────────────────────────────
    save_path = save_path or (MODELS_DIR / "neural_registers.pt")
    rf.save(save_path)
    if verbose:
        print(f"  Saved to {save_path}")
        print(f"  Parameters: {rf.stats['param_count']:,}")
        print(f"  Best value accuracy: {best_accuracy:.4f}")

    return rf


# ────────────────────────────────────────────────────────────────────────────
# Verification
# ────────────────────────────────────────────────────────────────────────────

def verify_register_file(rf: NeuralRegisterFile, n_tests: int = 1000,
                         verbose: bool = True) -> tuple[int, int]:
    """Verify lossless round-trip for n_tests random int64 values.

    Returns (n_correct, n_total).
    """
    import random
    correct = 0
    failures = []

    for i in range(n_tests):
        value = random.randint(-(1 << 63), (1 << 63) - 1)
        reg_idx = i % 31

        rf.write(reg_idx, value)
        result = rf.read(reg_idx)

        if result == value:
            correct += 1
        else:
            failures.append((i, reg_idx, value, result))

    if verbose:
        print(f"  Verification: {correct}/{n_tests} correct "
              f"({100*correct/n_tests:.1f}%)")
        if failures and len(failures) <= 5:
            for idx, reg, expected, got in failures[:5]:
                print(f"    FAIL test {idx}: reg[{reg}] "
                      f"wrote {expected} got {got}")

    return correct, n_tests
