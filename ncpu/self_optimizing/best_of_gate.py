"""Best-of-N over max_gate — safety net: worst case = baseline.

The idea: for each problem, generate a completion at several different
NPCoT `max_gate` values (including 0, which is baseline). Score each
candidate via a verifier or the model's own logprob. Return the best.

Guarantees:
- If `gate=0` is among the candidates, the worst-case result is the
  baseline (no NPCoT).
- If the library's contribution at some higher gate wins the verifier,
  we use it. Otherwise we fall back to baseline.

Contrast with FIX-5 (adaptive temperature): that varies sampling stochasticity.
This varies NPCoT's contribution strength. They compose: at library miss,
we can sample N times AND across M gates = N*M candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class BestOfGateConfig:
    """Which gates to try."""

    gate_values: tuple[float, ...] = (0.0, 0.02, 0.05)
    # If True, always include gate=0 even if not in gate_values. That's
    # the "never worse than baseline" guarantee.
    force_include_baseline: bool = True


@dataclass
class BestOfGateResult:
    """Outcome of the gate sweep."""

    selected_text: str
    selected_gate: float
    selected_logprob: float
    all_candidates: list[dict[str, Any]]


def select_best_over_gates(
    candidates: list[dict[str, Any]],
    *,
    verifier_fn: Optional[Callable[[str], float]] = None,
) -> BestOfGateResult:
    """Pick best candidate; guarantee baseline (gate=0) is a fallback.

    Each candidate dict: `{"text": str, "gate": float, "logprob": float,
    "verifier_score": Optional[float]}`.

    Selection priority:
    1. If `verifier_fn` given, max by verifier_score.
    2. Else, max by model logprob.
    3. Ties broken by lower gate (prefer baseline behavior).
    """
    if not candidates:
        raise ValueError("no candidates supplied")

    # Augment missing fields.
    for c in candidates:
        c.setdefault("gate", 0.0)
        c.setdefault("logprob", float("-inf"))
        c.setdefault("verifier_score", None)

    if verifier_fn is not None:
        for c in candidates:
            if c["verifier_score"] is None:
                c["verifier_score"] = verifier_fn(c["text"])
        scored = sorted(
            candidates,
            key=lambda c: (-c["verifier_score"], c["gate"]),
        )
    else:
        scored = sorted(
            candidates,
            key=lambda c: (-c["logprob"], c["gate"]),
        )
    best = scored[0]
    return BestOfGateResult(
        selected_text=best["text"],
        selected_gate=float(best["gate"]),
        selected_logprob=float(best["logprob"]),
        all_candidates=candidates,
    )


def set_coprocessor_gates(model, gate_value: float) -> None:
    """Set `config.max_gate` on every NCPUCoprocessorMLPWithArrayThought.

    This mutates in place. Call once per candidate generation with the
    desired gate, generate, then move on to the next gate.
    """
    for module in model.modules():
        if module.__class__.__name__ == "NCPUCoprocessorMLPWithArrayThought":
            if getattr(module, "array_thought", None) is not None:
                module.array_thought.config.max_gate = float(gate_value)


__all__ = [
    "BestOfGateConfig",
    "BestOfGateResult",
    "select_best_over_gates",
    "set_coprocessor_gates",
]
