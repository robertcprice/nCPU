"""Partial-code completion helpers for Mog.

This module supports a more realistic and easier synthesis task than full
all-mask generation: given a scaffold with function signatures / braces fixed,
generate the masked function bodies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from egdc.mog.tokenizer import MogCodeTokenizer, MASK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN


def mask_function_bodies(code: str, mask_char: str = " ") -> str:
    """Return a scaffold where function bodies are blanked out but signatures and
    braces are preserved.

    Heuristic: whenever we encounter a `{` that belongs to a `fn` declaration,
    mask all characters until the matching `}` except nested braces themselves.
    This keeps overall structure and line breaks while removing the body text.
    """
    chars = list(code)
    i = 0
    n = len(chars)
    while i < n:
        # Find a function declaration.
        if code.startswith("fn ", i) or code.startswith("pub fn ", i) or code.startswith("async fn ", i) or code.startswith("pub async fn ", i):
            # Advance to the opening brace for that function.
            j = i
            while j < n and chars[j] != "{":
                j += 1
            if j >= n:
                break
            depth = 1
            k = j + 1
            while k < n and depth > 0:
                if chars[k] == "{":
                    depth += 1
                elif chars[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                if depth > 0 and chars[k] not in "{}\n\r":
                    chars[k] = mask_char
                k += 1
            i = k + 1
            continue
        i += 1
    return "".join(chars)


def build_completion_tokens(code: str, tokenizer: MogCodeTokenizer, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build initial masked tokens, fixed-position mask, and original tokens.

    Fixed positions are all scaffold bytes (non-body bytes). Masked positions are
    all function-body bytes.
    """
    scaffold = mask_function_bodies(code)
    original = tokenizer.pad(tokenizer.encode(code), seq_len)
    scaffold_tokens = tokenizer.pad(tokenizer.encode(scaffold), seq_len)

    original_tokens = torch.tensor(original, dtype=torch.long)
    initial_tokens = torch.tensor(scaffold_tokens, dtype=torch.long)

    fixed_positions = torch.zeros(seq_len, dtype=torch.bool)
    for i, (orig_t, scaf_t) in enumerate(zip(original, scaffold_tokens)):
        if scaf_t in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
            fixed_positions[i] = True
        elif scaf_t == orig_t and chr(orig_t) != " ":
            fixed_positions[i] = True
        elif scaf_t == orig_t and orig_t not in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
            # preserve punctuation / braces / newlines too
            fixed_positions[i] = True
        else:
            initial_tokens[i] = MASK_TOKEN

    return initial_tokens, fixed_positions, original_tokens


@torch.no_grad()
def complete_mog_from_initial(
    model,
    initial_tokens: torch.Tensor,
    fixed_positions: torch.Tensor,
    spec_tokens: torch.Tensor | None,
    num_steps: int = 64,
    temperature: float = 0.8,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Iterative unmasking starting from a partially observed sequence.

    Positions marked in fixed_positions are never changed.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    tokens = initial_tokens.clone().to(device)
    fixed_positions = fixed_positions.to(device)
    if spec_tokens is not None:
        spec_tokens = spec_tokens.to(device)

    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if fixed_positions.ndim == 1:
        fixed_positions = fixed_positions.unsqueeze(0)

    seq_len = tokens.shape[1]

    for step in range(num_steps):
        t = 1.0 - (step + 1) / num_steps
        t_tensor = torch.tensor([max(t, 0.01)], device=device)
        logits = model(tokens, t_tensor, spec_tokens=spec_tokens)
        probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)

        flat_probs = probs.view(-1, probs.shape[-1]).clamp(min=1e-10)
        flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(flat_probs, num_samples=1).view(1, seq_len)

        confidence = probs.max(dim=-1).values
        is_masked = (tokens == MASK_TOKEN) & ~fixed_positions
        num_masked = is_masked.sum().item()
        if num_masked == 0:
            break

        num_to_reveal = max(1, min(num_masked, int((num_masked + max(num_steps - step, 1) - 1) / max(num_steps - step, 1))))
        masked_conf = confidence.clone()
        masked_conf[~is_masked] = -1.0
        _, top_indices = masked_conf.topk(num_to_reveal, dim=-1)
        for idx in top_indices[0]:
            tokens[0, idx] = sampled[0, idx]

    if (tokens == MASK_TOKEN).any():
        t_tensor = torch.tensor([0.01], device=device)
        logits = model(tokens, t_tensor, spec_tokens=spec_tokens)
        final = logits.argmax(dim=-1)
        mask = (tokens == MASK_TOKEN) & ~fixed_positions
        tokens = torch.where(mask, final, tokens)

    # Guarantee fixed positions are preserved.
    tokens = torch.where(fixed_positions, initial_tokens.to(device).view_as(tokens), tokens)
    return tokens
