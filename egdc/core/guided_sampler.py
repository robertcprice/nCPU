"""Execution-guided sampler using discrete execution for beam reranking.

At each denoising step, generate B candidate unmaskings and score them
by actual execution against test cases. Keep the best candidate.
This is more robust than gradient-based guidance because it uses the
real (non-differentiable) nCPU interpreter.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .model import MaskedDiffusionTransformer, MASK_TOKEN, PAD_TOKEN
from .sampler import build_slot_masks
from .evaluate import execute_program


def score_partial_program(
    tokens: List[int],
    test_cases: List[dict],
    max_steps: int = 200,
) -> float:
    """Score a (possibly partial) program by execution.

    Returns a score in [0, 1] where 1 = all tests pass.
    Partial credit for: halting, correct register direction, etc.
    """
    total_score = 0.0
    n_tests = len(test_cases)

    for tc in test_cases:
        inputs = tc.get("inputs", {})
        expected = tc.get("expected_output", 0)

        input_regs = {i: v for i, (_, v) in enumerate(inputs.items())}
        result = execute_program(tokens, input_regs, max_steps=max_steps)

        if result is None:
            # Doesn't halt: small score if it has HALT somewhere
            has_halt = 13 in tokens[::4]  # opcode HALT at slot 0 positions
            total_score += 0.05 if has_halt else 0.0
            continue

        # Halts: base credit
        total_score += 0.2

        actual = result.get(0, None)
        if actual is not None:
            if actual == expected:
                total_score += 0.8  # Full credit
            else:
                # Partial credit based on closeness
                if expected != 0:
                    rel_error = abs(actual - expected) / (abs(expected) + 1)
                    total_score += 0.3 * max(0, 1 - rel_error)
                elif actual == 0:
                    total_score += 0.3

    return total_score / max(n_tests, 1)


@torch.no_grad()
def guided_generate(
    model: MaskedDiffusionTransformer,
    spec_tokens: torch.Tensor,
    test_cases: List[dict],
    seq_len: int = 128,
    num_steps: int = 64,
    beam_width: int = 4,
    temperature: float = 0.3,
    guidance_start: float = 0.3,
    constrained: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, dict]:
    """Generate with execution-guided beam search during denoising.

    At each denoising step:
    1. Get model predictions for masked positions
    2. Sample B different unmaskings (different random draws)
    3. Score each by partial execution against test cases
    4. Keep the best-scoring candidate
    5. Continue denoising from the best candidate

    Args:
        model: trained diffusion model
        spec_tokens: (1, S) conditioning tokens
        test_cases: list of {inputs: {name: val}, expected_output: val}
        seq_len: output length
        num_steps: denoising steps
        beam_width: candidates per step
        temperature: sampling temperature
        guidance_start: fraction of steps before guidance kicks in
        constrained: enforce ISA slot constraints
        device: compute device

    Returns:
        (tokens, metrics)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    tokens = torch.full((1, seq_len), MASK_TOKEN, dtype=torch.long, device=device)
    spec_tokens = spec_tokens.to(device)
    slot_masks = build_slot_masks(seq_len).to(device) if constrained else None

    metrics = {"scores": [], "beam_used": 0}

    for step in range(num_steps):
        t = 1.0 - (step + 1) / num_steps
        t_tensor = torch.tensor([max(t, 0.01)], device=device)

        logits = model(tokens, t_tensor, spec_tokens=spec_tokens)[0]  # (L, V)

        if constrained and slot_masks is not None:
            logits = logits.clone()
            logits[~slot_masks] = -1e9

        probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)

        is_masked = (tokens[0] == MASK_TOKEN)
        num_masked = is_masked.sum().item()
        if num_masked == 0:
            break

        num_to_reveal = max(1, min(
            int(math.ceil(num_masked / max(num_steps - step, 1))),
            num_masked,
        ))

        # Confidence for position selection
        confidence = probs.max(dim=-1).values
        masked_conf = confidence.clone()
        masked_conf[~is_masked] = -1.0
        _, reveal_positions = masked_conf.topk(num_to_reveal)

        # Determine if we should use guidance this step
        progress = step / num_steps
        use_guidance = (progress >= guidance_start) and (num_masked < seq_len * 0.7)

        if use_guidance and beam_width > 1:
            # Generate B candidates
            best_tokens = None
            best_score = -1.0

            for b in range(beam_width):
                candidate = tokens.clone()
                # Sample tokens at reveal positions
                for pos in reveal_positions:
                    p = probs[pos].clamp(min=1e-10)
                    p = p / p.sum()
                    sampled = torch.multinomial(p.unsqueeze(0), 1).item()
                    candidate[0, pos] = sampled

                # Score by execution
                score = score_partial_program(
                    candidate[0].tolist(), test_cases, max_steps=100,
                )

                if score > best_score:
                    best_score = score
                    best_tokens = candidate

            tokens = best_tokens
            metrics["scores"].append(best_score)
            metrics["beam_used"] += 1
        else:
            # No guidance: just sample most confident
            flat_probs = probs.view(-1, probs.shape[-1]).clamp(min=1e-10)
            flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(flat_probs, num_samples=1).view(seq_len)
            for pos in reveal_positions:
                tokens[0, pos] = sampled[pos]

    # Final cleanup
    if (tokens == MASK_TOKEN).any():
        t_tensor = torch.tensor([0.01], device=device)
        logits = model(tokens, t_tensor, spec_tokens=spec_tokens)[0]
        if constrained and slot_masks is not None:
            logits = logits.clone()
            logits[~slot_masks] = -1e9
        probs = F.softmax(logits / temperature, dim=-1)
        flat_probs = probs.view(-1, probs.shape[-1]).clamp(min=1e-10)
        flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
        final = torch.multinomial(flat_probs, num_samples=1).view(1, seq_len)
        mask = (tokens == MASK_TOKEN)
        tokens = torch.where(mask, final, tokens)

    return tokens, metrics
