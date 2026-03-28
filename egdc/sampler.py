"""Inference sampler for masked diffusion.

Iteratively unmasks tokens from fully masked to fully unmasked,
using low-confidence remasking to refine predictions.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .model import MASK_TOKEN, MaskedDiffusionTransformer


@torch.no_grad()
def generate(
    model: MaskedDiffusionTransformer,
    spec_tokens: Optional[torch.Tensor],
    seq_len: int,
    num_steps: int = 64,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate a token sequence via iterative unmasking.

    Starts from all MASK tokens and progressively unmasks positions,
    keeping the most confident predictions at each step.

    Args:
        model: trained masked diffusion transformer
        spec_tokens: (1, S) conditioning spec tokens, or None
        seq_len: length of the sequence to generate
        num_steps: number of denoising steps
        temperature: sampling temperature (lower = more greedy)
        device: device to run on

    Returns:
        (1, seq_len) generated token IDs
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Start fully masked
    tokens = torch.full((1, seq_len), MASK_TOKEN, dtype=torch.long, device=device)

    if spec_tokens is not None:
        spec_tokens = spec_tokens.to(device)

    for step in range(num_steps):
        # Compute effective timestep: goes from ~1 (fully masked) to ~0 (clean)
        t = 1.0 - step / num_steps
        t_tensor = torch.tensor([t], device=device)

        # Get model predictions
        logits = model(tokens, t_tensor, spec_tokens=spec_tokens)  # (1, L, V)

        # Apply temperature and sample
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            # Sample from distribution
            flat_probs = probs.view(-1, probs.shape[-1])
            sampled = torch.multinomial(flat_probs, num_samples=1).view(1, seq_len)
        else:
            # Greedy: argmax
            sampled = logits.argmax(dim=-1)  # (1, L)

        # Compute confidence: max probability at each position
        with torch.no_grad():
            confidence = F.softmax(logits, dim=-1).max(dim=-1).values  # (1, L)

        # Determine how many positions to unmask at this step
        # Linear schedule: unmask proportionally more as steps progress
        fraction_to_unmask = (step + 1) / num_steps
        num_to_unmask = max(1, int(fraction_to_unmask * seq_len))
        num_to_unmask = min(num_to_unmask, seq_len)

        # Find currently masked positions
        is_masked = (tokens == MASK_TOKEN)  # (1, L)
        num_masked = is_masked.sum().item()

        if num_masked == 0:
            break

        # Among masked positions, pick the most confident to unmask
        # Set confidence of already-unmasked positions to -inf so we don't re-pick
        masked_confidence = confidence.clone()
        masked_confidence[~is_masked] = -1.0

        # Number to unmask this step (of remaining masked)
        num_to_reveal = max(1, min(
            int(math.ceil(num_masked * (1.0 / (num_steps - step)))),
            num_masked,
        ))

        # Get top-k most confident among masked positions
        _, top_indices = masked_confidence.topk(num_to_reveal, dim=-1)

        # Create new token tensor: keep unmasked, reveal top-k
        new_tokens = tokens.clone()
        new_tokens.scatter_(1, top_indices, sampled.gather(1, top_indices))
        tokens = new_tokens

    # Final pass at t≈0 to clean up any remaining masks
    if (tokens == MASK_TOKEN).any():
        t_tensor = torch.tensor([0.0], device=device)
        logits = model(tokens, t_tensor, spec_tokens=spec_tokens)
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            flat_probs = probs.view(-1, probs.shape[-1])
            final = torch.multinomial(flat_probs, num_samples=1).view(1, seq_len)
        else:
            final = logits.argmax(dim=-1)
        mask = (tokens == MASK_TOKEN)
        tokens = torch.where(mask, final, tokens)

    return tokens



