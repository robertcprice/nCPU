"""Entropy-driven adaptive sampling for NPCoT generation.

The idea: when the library doesn't know the current problem shape
(library miss or low hit confidence), the model is flying blind —
so *increase* exploration (higher temperature, more samples). When
the library confidently knows the shape, trust it — low temperature
greedy decode.

This converts "library miss" from a failure mode into an exploration
signal. The model now *tries more things* exactly when it has the least
cached guidance.

Two mechanisms compose:

1. **Adaptive temperature**: `τ(hidden) = τ_base + (1 - confidence) * τ_boost`.
   On a library hit → confidence=1 → τ = τ_base (usually 0, greedy).
   On a full miss → confidence=0 → τ = τ_base + τ_boost (exploratory).

2. **Best-of-N with verifier**: when library confidence < `explore_threshold`,
   generate N candidates and pick the one with highest model logprob
   (or with highest verifier score if a verifier is available).

Combined with FIX-4 (continual library growth), the system's behavior is:
- Unseen problem → high entropy → explore widely → verifier picks the best
  → successful solve → library grows → next time, low entropy → trust cache.

This gives the system a path to *eventually solve problems it has never
seen before* without needing additional training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


@dataclass
class AdaptiveSamplingConfig:
    """How to adapt sampling to library confidence."""

    temperature_base: float = 0.0          # when library is confident
    temperature_boost: float = 0.8         # added when library misses
    explore_threshold: float = 0.5         # below this confidence, sample N
    num_samples_on_miss: int = 5           # N for best-of-N
    num_samples_on_hit: int = 1            # greedy when confident


@dataclass
class AdaptiveSamplingResult:
    """What we sampled and why."""

    selected_text: str
    selected_logprob: float
    confidence: float                # library hit rate across tokens
    temperature_used: float
    num_candidates: int
    candidate_texts: list[str]
    candidate_logprobs: list[float]


def compute_confidence_from_hits(hit_indicators: list[bool]) -> float:
    """Confidence = fraction of tokens that hit the library.

    Simple but effective — if 80% of the tokens in the prefix have a
    library hit, we're fairly sure this problem matches something we've
    seen; 0% = totally unseen, explore.
    """
    if not hit_indicators:
        return 0.0
    return sum(1.0 for h in hit_indicators if h) / len(hit_indicators)


def adaptive_temperature(
    confidence: float, *, config: AdaptiveSamplingConfig
) -> float:
    """Map library confidence ∈ [0, 1] to a sampling temperature."""
    return config.temperature_base + (1.0 - confidence) * config.temperature_boost


def sequence_logprob(logprobs: torch.Tensor, generated_ids: torch.Tensor) -> float:
    """Sum of per-token logprobs for the generated tokens (mean-length-normalized).

    `logprobs[t, v]` is the model's log-probability of token v at position t.
    `generated_ids[t]` is the token actually generated.
    Returns mean logprob across the sequence (length-normalized so longer
    generations aren't unfairly penalized).
    """
    if generated_ids.numel() == 0:
        return 0.0
    picked = logprobs.gather(-1, generated_ids.unsqueeze(-1)).squeeze(-1)
    return float(picked.mean().item())


def select_best_candidate(
    candidates: list[dict[str, Any]],
    *,
    verifier_fn: Optional[Callable[[str], float]] = None,
) -> dict[str, Any]:
    """Pick the best candidate by verifier score or logprob.

    If `verifier_fn` is supplied, it scores each candidate's text → higher
    is better. Otherwise we use mean logprob (which correlates with model
    confidence in the generation).
    """
    if not candidates:
        raise ValueError("no candidates to select from")

    if verifier_fn is not None:
        scored = [(verifier_fn(c["text"]), c) for c in candidates]
        scored.sort(key=lambda s: s[0], reverse=True)
        return scored[0][1]

    return max(candidates, key=lambda c: c.get("logprob", float("-inf")))


def adaptive_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    hit_probe: Callable[[], list[bool]],
    config: AdaptiveSamplingConfig,
    max_new_tokens: int,
    device: str,
    verifier_fn: Optional[Callable[[str], float]] = None,
) -> AdaptiveSamplingResult:
    """Generate adaptively: more samples at higher temp when library misses.

    `hit_probe()` returns the list of per-token library hits for the most
    recent forward pass. Call after a dummy forward to probe confidence.
    (Passed as a callable so this function can stay model-agnostic.)
    """
    # First: dummy forward to probe library confidence.
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs)
    hits = hit_probe()
    confidence = compute_confidence_from_hits(hits)
    temp = adaptive_temperature(confidence, config=config)

    if confidence < config.explore_threshold:
        num_samples = config.num_samples_on_miss
    else:
        num_samples = config.num_samples_on_hit

    # Generate `num_samples` candidates.
    candidates: list[dict[str, Any]] = []
    for sample_idx in range(num_samples):
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": temp > 0.0,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if temp > 0.0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = 0.95
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = out.sequences[0][prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Compute logprob of the generation.
        if hasattr(out, "scores") and out.scores:
            stacked = torch.stack(out.scores, dim=0)  # (T, 1, V)
            logprobs = torch.log_softmax(stacked.squeeze(1), dim=-1)
            lp = sequence_logprob(logprobs, gen_ids[: logprobs.shape[0]])
        else:
            lp = 0.0
        candidates.append({"text": text, "logprob": lp, "index": sample_idx})

    best = select_best_candidate(candidates, verifier_fn=verifier_fn)
    return AdaptiveSamplingResult(
        selected_text=best["text"],
        selected_logprob=best["logprob"],
        confidence=confidence,
        temperature_used=temp,
        num_candidates=len(candidates),
        candidate_texts=[c["text"] for c in candidates],
        candidate_logprobs=[c["logprob"] for c in candidates],
    )


__all__ = [
    "AdaptiveSamplingConfig",
    "AdaptiveSamplingResult",
    "compute_confidence_from_hits",
    "adaptive_temperature",
    "sequence_logprob",
    "select_best_candidate",
    "adaptive_generate",
]
