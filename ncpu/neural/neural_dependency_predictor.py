"""Neural Dependency Graph Predictor v2 — targets 99%+ accuracy.

Single causal self-attention layer with 4 heads and 16-dim register embeddings.
The causal mask mirrors the lower-triangular hazard matrix. Attention heads
learn to match register indices (rd_j == rn_i for RAW hazards).

Architecture (~18K params):
  Input: [B, 5] — (rd, rn, rm, reads_rm, writes_rd) per instruction
  Encoder: 3× Embedding(33, 16) + 4 flags = 52-dim → Linear(52→64)
  Causal self-attention: 4 heads, causal mask
  FFN: Linear(64→128) → ReLU → Linear(128→64)
  Head: Linear(64→32) → ReLU → Linear(32→1) → sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralDependencyPredictor(nn.Module):
    def __init__(self, max_batch=64, d_model=64, n_heads=4, reg_dim=16):
        super().__init__()
        self.max_batch = max_batch
        self.d_model = d_model
        self.reg_embed = nn.Embedding(33, reg_dim)
        input_dim = reg_dim * 3 + 4  # 3 reg embeds + 4 flags
        self.inst_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model),
        )
        self.pos_embed = nn.Embedding(max_batch, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.hazard_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward_logits(self, rds, rns, rms, reads_rm, writes_rd):
        rd_emb = self.reg_embed(rds.clamp(0, 32).long())
        rn_emb = self.reg_embed(rns.clamp(0, 32).long())
        rm_emb = self.reg_embed(rms.clamp(0, 32).long())
        flags = torch.stack([
            reads_rm.float(), writes_rd.float(),
            (rds != 31).float(), torch.ones_like(reads_rm.float()),
        ], dim=-1)
        x = torch.cat([rd_emb, rn_emb, rm_emb, flags], dim=-1)
        x = self.inst_encoder(x)

        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)
        N, B, _ = x.shape
        device = x.device

        pos = self.pos_embed(torch.arange(B, device=device).clamp(0, self.max_batch - 1))
        x = x + pos
        causal = torch.triu(torch.ones(B, B, device=device, dtype=torch.bool), diagonal=1)
        attn_out, _ = self.attn(x, x, x, attn_mask=causal)
        x = self.attn_norm(x + attn_out)
        x = self.ffn_norm(x + self.ffn(x))

        logits = self.hazard_head(x).squeeze(-1)
        return logits.squeeze(0) if single else logits

    def forward(self, rds, rns, rms, reads_rm, writes_rd, BIG):
        logits = self.forward_logits(rds, rns, rms, reads_rm, writes_rd)
        B = rds.shape[-1] if rds.dim() > 1 else rds.shape[0]
        hazard_mask = torch.sigmoid(logits) > 0.3  # conservative: fewer missed hazards
        hazard_idx = torch.where(hazard_mask,
                                 torch.arange(B, device=rds.device, dtype=torch.int64), BIG)
        return hazard_idx.min()
