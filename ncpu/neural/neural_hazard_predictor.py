"""Neural Hazard Predictor — predicts read/write hazard properties from ARM64 instructions.

Replaces the deterministic bit-pattern hazard detection with a trained neural network,
demonstrating that even the CPU's hazard detection unit can be fully neural.

Architecture (~9K params, ~37KB):
  Input: 6 features extracted from 32-bit instruction encoding
  Hidden: 36→64→32→3 MLP with ReLU + Sigmoid
  Output: [reads_rn, reads_rm, writes_rd] probabilities
"""

import torch
import torch.nn as nn


class NeuralHazardPredictor(nn.Module):
    """Predicts per-instruction hazard properties from ARM64 instruction words.

    Input: instruction words [N] int64
    Output: [N, 3] float32 → [reads_rn, reads_rm, writes_rd] probabilities
    """

    def __init__(self, top_byte_dim=16, mid3_dim=8, funct_dim=8):
        super().__init__()
        self.top_byte_embed = nn.Embedding(256, top_byte_dim)
        self.mid3_embed = nn.Embedding(8, mid3_dim)
        self.funct_embed = nn.Embedding(64, funct_dim)
        input_dim = top_byte_dim + mid3_dim + funct_dim + 4  # sf + opc(2) + op_group
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )

    def extract_features(self, insts: torch.Tensor) -> torch.Tensor:
        """Extract instruction features as tensor ops (GPU-friendly)."""
        top_byte = ((insts >> 24) & 0xFF).long()
        mid3 = ((insts >> 21) & 0x7).long()
        funct = ((insts >> 10) & 0x3F).long()
        sf = ((insts >> 31) & 1).float().unsqueeze(-1)
        opc = ((insts >> 29) & 0x3)
        opc_lo = (opc & 1).float().unsqueeze(-1)
        opc_hi = ((opc >> 1) & 1).float().unsqueeze(-1)
        op_group = ((insts >> 24) & 1).float().unsqueeze(-1)

        return torch.cat([
            self.top_byte_embed(top_byte),
            self.mid3_embed(mid3),
            self.funct_embed(funct),
            sf, opc_lo, opc_hi, op_group,
        ], dim=-1)

    def forward(self, insts: torch.Tensor) -> torch.Tensor:
        """Predict hazard properties.

        Args:
            insts: [N] int64 instruction words

        Returns:
            [N, 3] float32 — [p_reads_rn, p_reads_rm, p_writes_rd]
        """
        features = self.extract_features(insts)
        return self.mlp(features)
