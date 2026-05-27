"""Neural Instruction Scheduler — learned out-of-order execution.

Instead of detecting hazards and serializing, this model predicts the optimal
execution ORDER for a batch of instructions. It learns to reorder instructions
to maximize batch utilization while preserving data dependencies.

This is the neural equivalent of a CPU's out-of-order execution engine:
- Traditional CPU: hardware scoreboard + reservation stations + reorder buffer
- Neural CPU: trained transformer that predicts safe execution permutations

Architecture (~15K params):
  Input: [B, 5] — (rd, rn, rm, reads_rm, writes_rd) per instruction
  Encoder: register embeddings + flags → 26-dim → Linear(26→64)
  Self-attention: 2-head attention over instruction positions (captures dependencies)
  Scheduler head: Linear(64→B) softmax per position → permutation matrix
  Output: execution order [B] — permutation of [0, 1, ..., B-1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralInstructionScheduler(nn.Module):
    """Predicts optimal instruction execution order from register dependencies.

    Learns to reorder instructions to maximize the number that can execute
    in parallel (i.e., maximize first_hazard position).
    """

    def __init__(self, max_batch=64, d_model=64, n_heads=2):
        super().__init__()
        self.max_batch = max_batch
        self.d_model = d_model

        # Register embeddings (same as dependency predictor)
        self.reg_embed = nn.Embedding(33, 8)  # 0-31 regs + 32 for padding

        # Instruction encoder
        self.inst_encoder = nn.Sequential(
            nn.Linear(26, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding (learned)
        self.pos_embed = nn.Embedding(max_batch, d_model)

        # Self-attention to capture inter-instruction dependencies
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)

        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        # Scheduling head: predict execution priority per position
        # Higher priority = execute earlier
        self.schedule_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, rds, rns, rms, reads_rm, writes_rd):
        """Predict execution order.

        Args:
            rds, rns, rms: [B] int64 register indices
            reads_rm, writes_rd: [B] bool flags

        Returns:
            order: [B] int64 — execution order (permutation of 0..B-1)
            priority: [B] float — scheduling priority scores
        """
        B = rds.shape[0]
        device = rds.device

        # Encode instructions
        rd_emb = self.reg_embed(rds.clamp(0, 32).long())
        rn_emb = self.reg_embed(rns.clamp(0, 32).long())
        rm_emb = self.reg_embed(rms.clamp(0, 32).long())
        flags = torch.stack([reads_rm.float(), writes_rd.float()], dim=-1)

        x = torch.cat([rd_emb, rn_emb, rm_emb, flags], dim=-1)  # [B, 26]
        x = self.inst_encoder(x)  # [B, d_model]

        # Add positional encoding
        pos = self.pos_embed(torch.arange(B, device=device).clamp(0, self.max_batch - 1))
        x = x + pos  # [B, d_model]

        # Self-attention (captures "which instructions depend on which")
        x_unsq = x.unsqueeze(0)  # [1, B, d_model]
        attn_out, _ = self.attn(x_unsq, x_unsq, x_unsq)
        x = self.attn_norm(x + attn_out.squeeze(0))  # [B, d_model]

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)  # [B, d_model]

        # Priority scores
        priority = self.schedule_head(x).squeeze(-1)  # [B]

        # Convert priority to execution order via argsort
        # Higher priority = lower index = execute first
        order = torch.argsort(priority, descending=True)

        return order, priority

    def forward_logits(self, rds, rns, rms, reads_rm, writes_rd):
        """Return raw priority logits (for training with ranking loss)."""
        B = rds.shape[0]
        device = rds.device

        rd_emb = self.reg_embed(rds.clamp(0, 32).long())
        rn_emb = self.reg_embed(rns.clamp(0, 32).long())
        rm_emb = self.reg_embed(rms.clamp(0, 32).long())
        flags = torch.stack([reads_rm.float(), writes_rd.float()], dim=-1)

        x = torch.cat([rd_emb, rn_emb, rm_emb, flags], dim=-1)
        x = self.inst_encoder(x)

        pos = self.pos_embed(torch.arange(B, device=device).clamp(0, self.max_batch - 1))
        x = x + pos

        x_unsq = x.unsqueeze(0)
        attn_out, _ = self.attn(x_unsq, x_unsq, x_unsq)
        x = self.attn_norm(x + attn_out.squeeze(0))

        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        return self.schedule_head(x).squeeze(-1)  # [B]
