"""Verifier-retry loop for code generation.

Standard pattern: try → verify → if fail, retry with different strategy.
Never worse than baseline (first attempt = baseline); usually better
(later attempts fix what first attempt broke).

Strategies cycle through:
    1. gate=0 (baseline, greedy)           — always first
    2. gate=0.02 (tiny NPCoT contribution) — "maybe the library helps"
    3. gate=0.05 (full NPCoT)              — "let NPCoT drive"
    4. gate=0 temp=0.5 (sample baseline)   — "same model, more variation"
    5. gate=0.05 temp=0.5 (sample NPCoT)   — "NPCoT + variation"

On each attempt the verifier scores the output. First candidate to pass
wins. If none pass, return the highest-scoring candidate (still better
than a guaranteed fail).

The retry loop is completely model-agnostic — pass it a `generate_fn` and
a `verify_fn`, it drives the sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class RetryStrategy:
    """One attempt's configuration."""

    gate: float
    temperature: float
    label: str


@dataclass
class RetryConfig:
    """Strategies ordered cheapest-first.

    Default schedule is the ablation-pruned 2-strategy pair from the full
    HumanEval run on Qwen3.5-4B (see paper 15.10): all observed retry-wins
    came from [gate=0.05, temp=0.5]; the intermediate greedy-NPCoT
    strategies rescued zero problems and were dropped.
    """

    strategies: list[RetryStrategy] = field(
        default_factory=lambda: [
            RetryStrategy(gate=0.0, temperature=0.0, label="baseline-greedy"),
            RetryStrategy(gate=0.05, temperature=0.5, label="npcot-sampled"),
        ]
    )
    stop_on_first_pass: bool = True
    max_attempts: int = 2


@dataclass
class RetryAttempt:
    """What happened on one attempt."""

    strategy_label: str
    gate: float
    temperature: float
    text: str
    verifier_passed: bool
    verifier_score: float
    error: Optional[str] = None


@dataclass
class RetryResult:
    """Outcome of the retry loop."""

    final_text: str
    final_passed: bool
    attempts: list[RetryAttempt]
    winning_attempt_index: Optional[int]
    total_attempts: int


def retry_until_verified(
    *,
    generate_fn: Callable[[RetryStrategy], str],
    verify_fn: Callable[[str], tuple[bool, float, Optional[str]]],
    config: Optional[RetryConfig] = None,
) -> RetryResult:
    """Drive the retry loop.

    `generate_fn(strategy) -> text`: produce a candidate under that strategy.
    `verify_fn(text) -> (passed, score, error)`: judge the candidate.

    Returns the first passing attempt, or the highest-scoring attempt if
    none pass.
    """
    cfg = config or RetryConfig()
    attempts: list[RetryAttempt] = []
    max_attempts = min(cfg.max_attempts, len(cfg.strategies))

    for idx in range(max_attempts):
        strat = cfg.strategies[idx]
        text = generate_fn(strat)
        passed, score, err = verify_fn(text)
        attempts.append(
            RetryAttempt(
                strategy_label=strat.label,
                gate=strat.gate,
                temperature=strat.temperature,
                text=text,
                verifier_passed=passed,
                verifier_score=score,
                error=err,
            )
        )
        if passed and cfg.stop_on_first_pass:
            return RetryResult(
                final_text=text,
                final_passed=True,
                attempts=attempts,
                winning_attempt_index=idx,
                total_attempts=len(attempts),
            )

    # None passed — return highest-scoring attempt.
    best_idx = max(
        range(len(attempts)),
        key=lambda i: (attempts[i].verifier_passed, attempts[i].verifier_score),
    )
    best = attempts[best_idx]
    return RetryResult(
        final_text=best.text,
        final_passed=best.verifier_passed,
        attempts=attempts,
        winning_attempt_index=best_idx if best.verifier_passed else None,
        total_attempts=len(attempts),
    )


__all__ = [
    "RetryStrategy",
    "RetryConfig",
    "RetryAttempt",
    "RetryResult",
    "retry_until_verified",
]
