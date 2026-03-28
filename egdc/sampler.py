"""Inference sampler for masked diffusion with constrained decoding.

Iteratively unmasks tokens from fully masked to fully unmasked,
using low-confidence remasking and ISA-aware token constraints.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .model import MASK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MaskedDiffusionTransformer
from .tokenizer import (
    NUM_OPCODES, OPCODE_OFFSET, REG_OFFSET, NUM_REGISTERS,
    IMM_OFFSET, NUM_IMMEDIATES, BR_OFFSET, NUM_BRANCH_TARGETS, VOCAB_SIZE,
)


def build_slot_masks(seq_len: int, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """Build per-position vocabulary masks enforcing ISA structure.

    Returns: (seq_len, vocab_size) boolean tensor. True = allowed token.

    Token layout per instruction (4 tokens):
      slot 0: opcode (tokens 0-13)
      slot 1: dst register (tokens 14-21)
      slot 2: src register (tokens 14-21)
      slot 3: immediate or branch target (tokens 22-341)

    BOS/EOS/PAD/MASK are always allowed (will be handled separately).
    """
    mask = torch.zeros(seq_len, vocab_size, dtype=torch.bool)

    for pos in range(seq_len):
        slot = pos % 4

        if slot == 0:
            # Opcode slot: allow opcodes
            mask[pos, OPCODE_OFFSET:OPCODE_OFFSET + NUM_OPCODES] = True
        elif slot == 1 or slot == 2:
            # Register slot: allow registers
            mask[pos, REG_OFFSET:REG_OFFSET + NUM_REGISTERS] = True
        elif slot == 3:
            # Immediate/branch slot: allow immediates and branch targets
            mask[pos, IMM_OFFSET:IMM_OFFSET + NUM_IMMEDIATES] = True
            mask[pos, BR_OFFSET:BR_OFFSET + NUM_BRANCH_TARGETS] = True

        # Always allow special tokens
        mask[pos, MASK_TOKEN] = True
        mask[pos, PAD_TOKEN] = True
        mask[pos, BOS_TOKEN] = True
        mask[pos, EOS_TOKEN] = True

    return mask


@torch.no_grad()
def generate(
    model: MaskedDiffusionTransformer,
    spec_tokens: Optional[torch.Tensor],
    seq_len: int,
    num_steps: int = 64,
    temperature: float = 0.8,
    device: Optional[torch.device] = None,
    constrained: bool = True,
) -> torch.Tensor:
    """Generate a token sequence via iterative unmasking.

    Args:
        model: trained masked diffusion transformer
        spec_tokens: (1, S) conditioning spec tokens, or None
        seq_len: length of the sequence to generate
        num_steps: number of denoising steps
        temperature: sampling temperature (lower = more greedy)
        device: device to run on
        constrained: if True, enforce ISA slot constraints on generated tokens

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

    # Build slot constraint masks
    if constrained:
        slot_masks = build_slot_masks(seq_len).to(device)  # (L, V)
        neg_inf = torch.tensor(float('-inf'), device=device)

    for step in range(num_steps):
        # Compute effective timestep: goes from ~1 (fully masked) to ~0 (clean)
        t = 1.0 - (step + 1) / num_steps
        t_tensor = torch.tensor([max(t, 0.01)], device=device)

        # Get model predictions
        logits = model(tokens, t_tensor, spec_tokens=spec_tokens)  # (1, L, V)

        # Apply slot constraints: mask out invalid tokens per position
        if constrained:
            logits = logits.clone()
            logits[0][~slot_masks] = -1e9

        # Apply temperature and compute probabilities
        probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)

        # Sample from distribution
        flat_probs = probs.view(-1, probs.shape[-1])
        # Clamp to avoid numerical issues
        flat_probs = flat_probs.clamp(min=1e-10)
        flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(flat_probs, num_samples=1).view(1, seq_len)

        # Compute confidence: max probability at each position
        confidence = probs.max(dim=-1).values  # (1, L)

        # Find currently masked positions
        is_masked = (tokens == MASK_TOKEN)  # (1, L)
        num_masked = is_masked.sum().item()

        if num_masked == 0:
            break

        # How many to unmask this step
        num_to_reveal = max(1, min(
            int(math.ceil(num_masked / max(num_steps - step, 1))),
            num_masked,
        ))

        # Among masked positions, pick the most confident to unmask
        masked_confidence = confidence.clone()
        masked_confidence[~is_masked] = -1.0

        _, top_indices = masked_confidence.topk(num_to_reveal, dim=-1)

        # Reveal those positions
        for idx in top_indices[0]:
            tokens[0, idx] = sampled[0, idx]

    # Final cleanup: replace any remaining masks
    if (tokens == MASK_TOKEN).any():
        t_tensor = torch.tensor([0.01], device=device)
        logits = model(tokens, t_tensor, spec_tokens=spec_tokens)
        if constrained:
            logits = logits.clone()
            logits[0][~slot_masks] = -1e9
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            flat_probs = probs.view(-1, probs.shape[-1]).clamp(min=1e-10)
            flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
            final = torch.multinomial(flat_probs, num_samples=1).view(1, seq_len)
        else:
            final = logits.argmax(dim=-1)
        mask = (tokens == MASK_TOKEN)
        tokens = torch.where(mask, final, tokens)

    return tokens
