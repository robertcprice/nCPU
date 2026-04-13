"""Python code masked diffusion transformer.

Adapts the base MaskedDiffusionTransformer for Python code generation:
- Byte-level vocab (260 tokens) instead of nCPU ISA (346 tokens)
- Longer sequences (1024) for Python programs
- No instruction slot embeddings (Python isn't 4-token structured)
- Character-position embeddings instead
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from egdc.core.model import (
    MaskedDiffusionTransformer,
    ModelConfig,
    TimestepEmbedder,
    LearnedPositionalEncoding,
)
from egdc.python.tokenizer import VOCAB_SIZE, MASK_TOKEN, PAD_TOKEN


@dataclass
class PythonDiffusionConfig(ModelConfig):
    """Configuration for Python code masked diffusion transformer.

    Overrides nCPU defaults for Python code generation:
    - Smaller hidden_dim (512) since byte-level tokens are simpler
    - 8 layers/heads instead of 12
    - Longer max_seq_len (1024) for Python programs
    - 260-token byte-level vocabulary
    """
    vocab_size: int = VOCAB_SIZE     # 260 (256 bytes + 4 special)
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ff_dim: int = 2048               # 4x hidden_dim
    max_seq_len: int = 1024          # Python programs are longer
    dropout: float = 0.1
    timestep_dim: int = 256

    @classmethod
    def tiny(cls) -> "PythonDiffusionConfig":
        return cls(hidden_dim=256, num_layers=4, num_heads=4, ff_dim=1024, timestep_dim=64)

    @classmethod
    def small(cls) -> "PythonDiffusionConfig":
        return cls(hidden_dim=384, num_layers=6, num_heads=6, ff_dim=1536, timestep_dim=128)

    @classmethod
    def medium(cls) -> "PythonDiffusionConfig":
        return cls()  # defaults are medium


class PythonMaskedDiffusion(nn.Module):
    """Masked diffusion transformer for Python code generation.

    Built from scratch (not subclassing MaskedDiffusionTransformer) to avoid
    the slot embedding and nCPU-specific structure. Uses the same architectural
    components (transformer encoder, timestep embedding, positional encoding)
    but configured for byte-level Python code.
    """

    def __init__(self, config: Optional[PythonDiffusionConfig] = None) -> None:
        super().__init__()
        self.config = config or PythonDiffusionConfig()
        c = self.config

        # Token and position embeddings
        self.token_embed = nn.Embedding(c.vocab_size, c.hidden_dim)
        self.pos_embed = LearnedPositionalEncoding(c.max_seq_len, c.hidden_dim)

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
        code_emb = code_emb + self.segment_embed(
            torch.ones(1, dtype=torch.long, device=device)
        )

        # No slot embedding - Python isn't 4-token structured

        # Add timestep embedding to every position (broadcast)
        t_emb = self.timestep_embed(timesteps)  # (B, H)
        code_emb = code_emb + t_emb.unsqueeze(1)

        # Prepend spec tokens as prefix if provided
        if spec_tokens is not None:
            S = spec_tokens.shape[1]
            spec_emb = self.token_embed(spec_tokens)  # (B, S, H)
            spec_emb = spec_emb + self.pos_embed(S, device)[:, :S]
            spec_emb = spec_emb + self.segment_embed(
                torch.zeros(1, dtype=torch.long, device=device)
            )
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
