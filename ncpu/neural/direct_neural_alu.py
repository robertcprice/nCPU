"""Direct Neural ALU: single-pass MLP that performs 32-bit arithmetic.

Replaces the Kogge-Stone CLA (160 sequential MLP evaluations per ADD) with
ONE forward pass through a trained MLP:

    (a_bits[32], b_bits[32], op_onehot[8]) -> (result_bits[32], flags[4])

Supported operations: ADD, SUB, AND, OR, XOR (indices 0-4).
Op slots 5-7 are reserved for future use (MUL_LOW, SHL, SHR).

The architecture uses residual blocks with layer normalization for stable
training on the hard carry-propagation problem (e.g. 0xFFFFFFFF + 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the GPU-native bit conversion from neural_weave
from ncpu.neural.neural_weave import bits_to_int_tensor, int_tensor_to_bits

# ─── Op encoding ─────────────────────────────────────────────────────────────

OP_NAMES = ["add", "sub", "and", "or", "xor"]
OP_TO_IDX = {name: i for i, name in enumerate(OP_NAMES)}
NUM_OP_CLASSES = 8  # 5 used + 3 reserved


# ─── Model architecture ─────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Pre-norm residual block: LN -> Linear -> GELU -> Linear -> add."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class DirectNeuralALU(nn.Module):
    """Single-pass neural ALU: (a, b, op) -> (result, flags) in one forward pass.

    Input:  a_bits(32) + b_bits(32) + op_onehot(8) = 72
    Output: result_bits(32) + N,Z,C,V flags(4) = 36

    Architecture: projection -> N residual blocks -> output heads.
    """

    def __init__(self, hidden: int = 512, n_blocks: int = 4):
        super().__init__()
        self.hidden = hidden
        self.n_blocks = n_blocks

        # Input projection
        self.input_proj = nn.Linear(72, hidden)

        # Residual tower
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_blocks)])

        # Separate output heads for result bits vs flags
        self.result_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 32),
        )
        self.flags_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

    def forward(
        self, a_bits: torch.Tensor, b_bits: torch.Tensor, op_code: torch.Tensor
    ) -> torch.Tensor:
        """
        a_bits:  (batch, 32) float, each element 0.0 or 1.0, LSB first
        b_bits:  (batch, 32) float, each element 0.0 or 1.0, LSB first
        op_code: (batch, 8) float one-hot encoding
        Returns: (batch, 36) -- result_bits(32) + N,Z,C,V flags(4)
        """
        x = torch.cat([a_bits, b_bits, op_code], dim=1)  # (batch, 72)
        h = F.gelu(self.input_proj(x))

        for block in self.blocks:
            h = block(h)

        result_logits = self.result_head(h)  # (batch, 32)
        flag_logits = self.flags_head(h)  # (batch, 4)

        return torch.cat([result_logits, flag_logits], dim=1)  # (batch, 36)


# ─── Inference wrapper ───────────────────────────────────────────────────────

class DirectALU:
    """Fast neural ALU: one forward pass per operation.

    Drop-in replacement for the Kogge-Stone CLA path. Loads the trained
    direct_alu.pt checkpoint and provides scalar + batched execution.
    """

    DEFAULT_PATH = Path("models/alu/direct_alu.pt")

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "mps",
        hidden: int = 512,
        n_blocks: int = 4,
    ):
        path = Path(model_path) if model_path else self.DEFAULT_PATH

        self.device = torch.device(device)
        self.model = DirectNeuralALU(hidden=hidden, n_blocks=n_blocks)

        state = torch.load(path, map_location=self.device, weights_only=True)
        # Support both raw state_dict and wrapped checkpoint
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        # Precompute powers-of-two for bits->int
        self._pow2 = (1 << torch.arange(32, dtype=torch.int64, device=self.device)).float()

    @torch.no_grad()
    def execute(self, a: int, b: int, op: str) -> tuple[int, dict]:
        """Scalar execution. Returns (result, {N, Z, C, V})."""
        a_t = torch.tensor([a & 0xFFFFFFFF], dtype=torch.int64, device=self.device)
        b_t = torch.tensor([b & 0xFFFFFFFF], dtype=torch.int64, device=self.device)

        a_bits = int_tensor_to_bits(a_t, 32)
        b_bits = int_tensor_to_bits(b_t, 32)

        op_idx = OP_TO_IDX[op.lower()]
        op_oh = F.one_hot(
            torch.tensor([op_idx], device=self.device), num_classes=NUM_OP_CLASSES
        ).float()

        out = self.model(a_bits, b_bits, op_oh)  # (1, 36)
        result_bits = (out[0, :32] > 0.0).float()  # logit threshold at 0
        flag_bits = (out[0, 32:] > 0.0).float()

        result_val = bits_to_int_tensor(result_bits.unsqueeze(0))
        result_int = int(result_val.item()) & 0xFFFFFFFF

        # Sign-extend to signed 32-bit for consistency
        if result_int >= 0x80000000:
            result_int -= 0x100000000

        flags = {
            "N": bool(flag_bits[0] > 0.5),
            "Z": bool(flag_bits[1] > 0.5),
            "C": bool(flag_bits[2] > 0.5),
            "V": bool(flag_bits[3] > 0.5),
        }
        return result_int, flags

    @torch.no_grad()
    def execute_batch(
        self, a: torch.Tensor, b: torch.Tensor, ops: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched execution.

        Args:
            a: (N,) int64 tensor of 32-bit unsigned values
            b: (N,) int64 tensor of 32-bit unsigned values
            ops: (N,) int64 tensor of op indices (0=ADD, 1=SUB, 2=AND, 3=OR, 4=XOR)

        Returns:
            results: (N,) int64 tensor of 32-bit results
            flags: (N, 4) float tensor [N, Z, C, V]
        """
        a_bits = int_tensor_to_bits(a.to(self.device), 32)
        b_bits = int_tensor_to_bits(b.to(self.device), 32)
        op_oh = F.one_hot(ops.to(self.device), num_classes=NUM_OP_CLASSES).float()

        out = self.model(a_bits, b_bits, op_oh)  # (N, 36)
        result_bits = (out[:, :32] > 0.0).float()
        flag_probs = (out[:, 32:] > 0.0).float()

        results = bits_to_int_tensor(result_bits)
        return results, flag_probs

    def export_flat_weights(self, path: str) -> None:
        """Export all model weights as a flat float32 buffer for Metal shader.

        Format: each parameter flattened and concatenated in state_dict order.
        A companion .json file records the layout (name, shape, offset, count).
        """
        import json

        params = []
        offset = 0
        flat_parts = []

        for name, param in self.model.state_dict().items():
            flat = param.detach().cpu().float().flatten()
            params.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "offset": offset,
                    "count": flat.numel(),
                }
            )
            flat_parts.append(flat)
            offset += flat.numel()

        flat_all = torch.cat(flat_parts)
        flat_all.numpy().tofile(path)

        meta_path = path.replace(".bin", ".json")
        with open(meta_path, "w") as f:
            json.dump(
                {"total_floats": offset, "parameters": params},
                f,
                indent=2,
            )
