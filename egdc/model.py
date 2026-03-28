"""Masked Diffusion Transformer for EGDC.

Encoder-only transformer that takes partially masked token sequences
and predicts the original tokens at masked positions.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Token constants
VOCAB_SIZE = 346
MASK_TOKEN = 342
PAD_TOKEN = 343
BOS_TOKEN = 344
EOS_TOKEN = 345


@dataclass
class ModelConfig:
    """Configuration for the masked diffusion transformer."""
    vocab_size: int = VOCAB_SIZE
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ff_dim: int = 3072  # 4x hidden_dim
    max_seq_len: int = 512
    dropout: float = 0.1
    timestep_dim: int = 256

    @classmethod
    def tiny(cls) -> "ModelConfig":
        return cls(hidden_dim=256, num_layers=4, num_heads=4, ff_dim=1024, timestep_dim=64)

    @classmethod
    def small(cls) -> "ModelConfig":
        return cls(hidden_dim=384, num_layers=6, num_heads=6, ff_dim=1536, timestep_dim=128)

    @classmethod
    def medium(cls) -> "ModelConfig":
        return cls()  # defaults are medium


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for continuous timestep t in [0, 1]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float tensor in [0,1] -> (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal + MLP timestep embedding."""

    def __init__(self, hidden_dim: int, timestep_dim: int = 256) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbedding(timestep_dim)
        self.mlp = nn.Sequential(
            nn.Linear(timestep_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> (B, hidden_dim)"""
        return self.mlp(self.sinusoidal(t))


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding."""

    def __init__(self, max_seq_len: int, hidden_dim: int) -> None:
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Returns (1, seq_len, hidden_dim)"""
        positions = torch.arange(seq_len, device=device)
        return self.pos_embed(positions).unsqueeze(0)


class MaskedDiffusionTransformer(nn.Module):
    """Encoder-only transformer for masked diffusion language modeling.

    Takes token sequences (some positions masked) plus a continuous timestep,
    and predicts logits over the vocabulary at every position.
    Spec tokens are prepended as a prefix for conditioning.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config

        # Token and position embeddings
        self.token_embed = nn.Embedding(c.vocab_size, c.hidden_dim)
        self.pos_embed = LearnedPositionalEncoding(c.max_seq_len, c.hidden_dim)

        # Instruction slot embedding: position % 4 -> {opcode, dst, src, imm} slot
        self.slot_embed = nn.Embedding(4, c.hidden_dim)

        # Timestep conditioning
        self.timestep_embed = TimestepEmbedder(c.hidden_dim, c.timestep_dim)

        # Segment embedding: 0 = spec prefix, 1 = code sequence
        self.segment_embed = nn.Embedding(2, c.hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.hidden_dim,
            nhead=c.num_heads,
            dim_feedforward=c.ff_dim,
            dropout=c.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=c.num_layers,
            norm=nn.LayerNorm(c.hidden_dim),
        )

        # Output head
        self.output_norm = nn.LayerNorm(c.hidden_dim)
        self.output_proj = nn.Linear(c.hidden_dim, c.vocab_size, bias=False)

        # Weight tying: share token embedding and output projection
        self.output_proj.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        timesteps: torch.Tensor,
        spec_tokens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: (B, L) token IDs, some positions set to MASK_TOKEN
            timesteps: (B,) continuous timesteps in [0, 1]
            spec_tokens: (B, S) optional spec/conditioning token IDs
            padding_mask: (B, L) bool, True = padded (ignored) positions

        Returns:
            logits: (B, L, vocab_size) predictions at each code position
        """
        B, L = token_ids.shape
        device = token_ids.device

        # Embed code tokens
        code_emb = self.token_embed(token_ids)  # (B, L, H)
        code_emb = code_emb + self.pos_embed(L, device)[:, :L]
        code_emb = code_emb + self.segment_embed(torch.ones(1, dtype=torch.long, device=device))

        # Add instruction slot embeddings (position % 4)
        slot_ids = torch.arange(L, device=device) % 4
        code_emb = code_emb + self.slot_embed(slot_ids).unsqueeze(0)

        # Add timestep embedding to every position (broadcast)
        t_emb = self.timestep_embed(timesteps)  # (B, H)
        code_emb = code_emb + t_emb.unsqueeze(1)

        # Prepend spec tokens as prefix if provided
        if spec_tokens is not None:
            S = spec_tokens.shape[1]
            spec_emb = self.token_embed(spec_tokens)  # (B, S, H)
            spec_emb = spec_emb + self.pos_embed(S, device)[:, :S]
            spec_emb = spec_emb + self.segment_embed(torch.zeros(1, dtype=torch.long, device=device))
            spec_emb = spec_emb + t_emb.unsqueeze(1)

            # Concatenate: [spec | code]
            x = torch.cat([spec_emb, code_emb], dim=1)  # (B, S+L, H)

            # Extend padding mask
            if padding_mask is not None:
                spec_pad = torch.zeros(B, S, dtype=torch.bool, device=device)
                padding_mask = torch.cat([spec_pad, padding_mask], dim=1)
        else:
            x = code_emb
            S = 0

        # Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Extract code positions only (drop spec prefix)
        x = x[:, S:]  # (B, L, H)

        # Project to vocab
        logits = self.output_proj(self.output_norm(x))  # (B, L, vocab_size)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
