"""Masking noise schedule for masked diffusion.

Cosine schedule: at timestep t in [0,1], the masking probability is
    beta(t) = 0.5 * (1 - cos(pi * t))

At t=0, beta=0 (no masking). At t=1, beta=1 (fully masked).
"""

import math
from typing import Optional

import torch


MASK_TOKEN = 342


def cosine_masking_rate(t: torch.Tensor) -> torch.Tensor:
    """Cosine masking schedule.

    Args:
        t: timesteps in [0, 1], any shape

    Returns:
        masking probabilities, same shape as t
    """
    return 0.5 * (1.0 - torch.cos(math.pi * t))


def sample_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample uniform random timesteps in [0, 1].

    Args:
        batch_size: number of timesteps to sample
        device: torch device

    Returns:
        (batch_size,) tensor of timesteps
    """
    return torch.rand(batch_size, device=device)


def get_mask(
    seq_len: int,
    t: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate random binary masks with masking rate beta(t).

    Args:
        seq_len: length of the sequence
        t: (B,) timesteps in [0, 1]
        device: torch device (defaults to t.device)

    Returns:
        (B, seq_len) bool tensor, True = position should be masked
    """
    if device is None:
        device = t.device
    B = t.shape[0]
    rates = cosine_masking_rate(t)  # (B,)
    rand = torch.rand(B, seq_len, device=device)
    return rand < rates.unsqueeze(1)


def apply_mask(
    token_ids: torch.Tensor,
    mask: torch.Tensor,
    mask_token_id: int = MASK_TOKEN,
) -> torch.Tensor:
    """Apply a mask to token IDs, replacing masked positions with MASK token.

    Args:
        token_ids: (B, L) original token IDs
        mask: (B, L) bool tensor, True = mask this position
        mask_token_id: ID of the mask token

    Returns:
        (B, L) token IDs with masked positions replaced
    """
    return torch.where(mask, mask_token_id, token_ids)
