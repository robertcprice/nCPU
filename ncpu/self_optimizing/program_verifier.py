"""Static analyzer for `DiscreteArrayProgram` (NV1).

Every cached skill in an `ArrayProgramLibrary` is a 5-tuple
`(init_idx, transform_idx, reduce_idx, post_scale_idx, offset)`. Because
the program is tiny, finite, and side-effect free, a complete static
analysis is straightforward and always terminates. This module proves:

* **Termination** — every program finishes in O(L) ops (trivially true
  given the language has no unbounded loops). We still emit the formal
  certificate for downstream audit pipelines that require it.
* **Division safety** — library programs never divide by zero. The only
  division is `acc / max(len(arr), 1)` in the `acc/len` post-scale, which
  is guarded by the clamp. We certify this explicitly.
* **Numerical overflow risk** — flags programs whose `reduce=*` on large
  inputs or whose `post_scale=exp(acc)` without log-domain operands will
  blow up float32. The verifier reports a *risk level* (safe / warn /
  high) rather than rejecting the program, because some callers (e.g.
  differentiable training) tolerate overflow via clamping.
* **Input-range sensitivity** — estimates the output's |value| bound given
  a caller-supplied bound on |input array element| and array length. Useful
  for regulated-workflow preflight checks.

Every verification result is a machine-readable `VerificationReport`
dataclass with a one-field-per-claim structure, suitable for rendering
into a compliance audit trail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ncpu.self_optimizing.array_executable_thought_head import (
    _ELEM_TRANSFORMS,
    _INIT_CHOICES,
    _POST_SCALES,
    _REDUCE_OPS,
)
from ncpu.self_optimizing.array_program_library import (
    DiscreteArrayProgram,
    LibraryEntry,
    _INIT_VALUES,
)


# ---------------------------------------------------------------------------
# Risk taxonomy
# ---------------------------------------------------------------------------

RISK_SAFE = "safe"
RISK_WARN = "warn"
RISK_HIGH = "high"


@dataclass
class RangeBound:
    """Conservative lower/upper bound on a scalar value."""

    lower: float
    upper: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": float(self.lower),
            "upper": float(self.upper),
        }

    def width(self) -> float:
        return abs(self.upper - self.lower)


@dataclass
class VerificationClaim:
    """One proven or rejected property about the program."""

    name: str
    verdict: bool
    risk_level: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    """Complete safety analysis of a single `DiscreteArrayProgram`."""

    program: dict[str, Any]
    claims: list[VerificationClaim] = field(default_factory=list)
    output_bound: Optional[RangeBound] = None
    worst_risk: str = RISK_SAFE
    overall_safe: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": dict(self.program),
            "claims": [
                {
                    "name": claim.name,
                    "verdict": bool(claim.verdict),
                    "risk_level": claim.risk_level,
                    "message": claim.message,
                    "evidence": dict(claim.evidence),
                }
                for claim in self.claims
            ],
            "output_bound": (
                self.output_bound.to_dict()
                if self.output_bound is not None
                else None
            ),
            "worst_risk": self.worst_risk,
            "overall_safe": bool(self.overall_safe),
        }


# ---------------------------------------------------------------------------
# Transform bounds
# ---------------------------------------------------------------------------

def _transform_bounds(
    transform_idx: int, input_bound: RangeBound
) -> RangeBound:
    if transform_idx == 0:  # x
        return input_bound
    if transform_idx == 1:  # x^2
        max_sq = max(input_bound.upper**2, input_bound.lower**2)
        min_sq = 0.0 if (input_bound.lower <= 0 <= input_bound.upper) else min(
            input_bound.upper**2, input_bound.lower**2
        )
        return RangeBound(min_sq, max_sq)
    if transform_idx == 2:  # |x|
        upper = max(abs(input_bound.lower), abs(input_bound.upper))
        lower = 0.0 if (input_bound.lower <= 0 <= input_bound.upper) else min(
            abs(input_bound.lower), abs(input_bound.upper)
        )
        return RangeBound(lower, upper)
    if transform_idx == 3:  # 1
        return RangeBound(1.0, 1.0)
    if transform_idx == 4:  # 1{x>0}
        return RangeBound(0.0, 1.0)
    if transform_idx == 5:  # log(|x|+eps)
        from ncpu.self_optimizing.array_executable_thought_head import (
            _LOG_EPS,
        )

        upper_abs = max(abs(input_bound.lower), abs(input_bound.upper))
        lower_abs = 0.0 if (input_bound.lower <= 0 <= input_bound.upper) else min(
            abs(input_bound.lower), abs(input_bound.upper)
        )
        import math

        lower = math.log(lower_abs + _LOG_EPS)
        upper = math.log(upper_abs + _LOG_EPS)
        return RangeBound(lower, upper)
    raise ValueError(f"unknown transform {transform_idx}")


def _reduce_fold_bounds(
    initial: float,
    per_element: RangeBound,
    reduce_idx: int,
    max_length: int,
) -> RangeBound:
    """Conservatively bound the accumulator after a length-L fold."""
    if reduce_idx == 0:  # +
        lower = initial + per_element.lower * max_length
        upper = initial + per_element.upper * max_length
        return RangeBound(lower, upper)
    if reduce_idx == 1:  # *
        # Worst-case expansion; sign-insensitive.
        max_abs_per = max(abs(per_element.lower), abs(per_element.upper))
        magnitude = abs(initial) * (max_abs_per**max_length)
        return RangeBound(-magnitude, magnitude)
    if reduce_idx == 2:  # max
        upper = max(initial, per_element.upper)
        lower = max(initial, per_element.lower)
        return RangeBound(lower, upper)
    if reduce_idx == 3:  # min
        upper = min(initial, per_element.upper)
        lower = min(initial, per_element.lower)
        return RangeBound(lower, upper)
    raise ValueError(f"unknown reduce {reduce_idx}")


def _post_scale_bounds(
    acc_bound: RangeBound,
    post_scale_idx: int,
    max_length: int,
) -> RangeBound:
    if post_scale_idx == 0:
        return acc_bound
    if post_scale_idx == 1:  # acc / len
        denom = max(1, max_length)
        return RangeBound(acc_bound.lower / denom, acc_bound.upper / denom)
    if post_scale_idx == 2:  # exp(clamp(acc, -30, 30))
        import math

        lower_clamped = max(acc_bound.lower, -30.0)
        upper_clamped = min(acc_bound.upper, 30.0)
        return RangeBound(math.exp(lower_clamped), math.exp(upper_clamped))
    raise ValueError(f"unknown post_scale {post_scale_idx}")


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

@dataclass
class VerifierConfig:
    """Assumptions the verifier uses to bound a program's output."""

    input_bound: RangeBound = field(
        default_factory=lambda: RangeBound(-10.0, 10.0)
    )
    max_length: int = 16
    overflow_threshold: float = 1e6


def verify_program(
    program: DiscreteArrayProgram,
    config: Optional[VerifierConfig] = None,
) -> VerificationReport:
    """Return a full `VerificationReport` for one discrete program."""
    cfg = config or VerifierConfig()
    report = VerificationReport(program=program.to_dict())

    # --- Termination ---
    report.claims.append(
        VerificationClaim(
            name="termination",
            verdict=True,
            risk_level=RISK_SAFE,
            message=(
                "Program is a bounded O(L) loop over a fixed-length array. "
                "No branches, no unbounded recursion; trivially terminates."
            ),
            evidence={"max_length_assumed": cfg.max_length},
        )
    )

    # --- Side-effect freedom ---
    report.claims.append(
        VerificationClaim(
            name="side_effect_free",
            verdict=True,
            risk_level=RISK_SAFE,
            message=(
                "Pure function over scalar accumulator; no writes to global "
                "state, no IO, no allocation."
            ),
        )
    )

    # --- Division safety ---
    if program.post_scale_idx == 1:
        # acc / max(len, 1) — guarded by clamp.
        report.claims.append(
            VerificationClaim(
                name="division_safety",
                verdict=True,
                risk_level=RISK_SAFE,
                message=(
                    "post_scale=acc/len uses max(len, 1) to avoid div-by-zero."
                ),
                evidence={"post_scale": _POST_SCALES[program.post_scale_idx]},
            )
        )
    else:
        report.claims.append(
            VerificationClaim(
                name="division_safety",
                verdict=True,
                risk_level=RISK_SAFE,
                message="Program has no division.",
            )
        )

    # --- Range analysis & overflow risk ---
    init_val = _INIT_VALUES[program.init_idx]
    per_elem = _transform_bounds(program.transform_idx, cfg.input_bound)
    acc_bound = _reduce_fold_bounds(
        init_val, per_elem, program.reduce_idx, cfg.max_length
    )
    post_bound = _post_scale_bounds(
        acc_bound, program.post_scale_idx, cfg.max_length
    )
    # Pre-offset bound (acc after post_scale but before offset addition).
    pre_offset_bound = RangeBound(post_bound.lower, post_bound.upper)
    final_lower = post_bound.lower + program.offset
    final_upper = post_bound.upper + program.offset
    report.output_bound = RangeBound(final_lower, final_upper)

    # Record the offset's contribution as a first-class claim so auditors
    # can reason about it independently. A program with a tiny offset is
    # safer to trust than one whose offset dominates the accumulator.
    offset_dominates = (
        abs(program.offset) > max(1.0, pre_offset_bound.width())
    )
    report.claims.append(
        VerificationClaim(
            name="offset_magnitude",
            verdict=not offset_dominates,
            risk_level=RISK_WARN if offset_dominates else RISK_SAFE,
            message=(
                f"Offset={program.offset:+.4f}; pre-offset accumulator "
                f"range width={pre_offset_bound.width():.4f}. Offset "
                f"{'dominates' if offset_dominates else 'does not dominate'} "
                "the accumulator."
            ),
            evidence={
                "offset": float(program.offset),
                "pre_offset_bound": pre_offset_bound.to_dict(),
                "post_offset_bound": report.output_bound.to_dict(),
            },
        )
    )

    magnitude = max(abs(final_lower), abs(final_upper))
    if magnitude > cfg.overflow_threshold:
        overflow_risk = RISK_HIGH
        overflow_verdict = False
    elif magnitude > cfg.overflow_threshold / 100.0:
        overflow_risk = RISK_WARN
        overflow_verdict = True
    else:
        overflow_risk = RISK_SAFE
        overflow_verdict = True

    report.claims.append(
        VerificationClaim(
            name="overflow_risk",
            verdict=overflow_verdict,
            risk_level=overflow_risk,
            message=(
                f"Output magnitude is bounded by |{magnitude:.3g}| given "
                f"input in [{cfg.input_bound.lower:.3g}, "
                f"{cfg.input_bound.upper:.3g}], length ≤ {cfg.max_length}. "
                f"Threshold = {cfg.overflow_threshold:.3g}."
            ),
            evidence={
                "output_bound": report.output_bound.to_dict(),
                "input_bound": cfg.input_bound.to_dict(),
                "max_length": cfg.max_length,
            },
        )
    )

    # --- Structural cautions (heuristic, not a rejection) ---
    product_risk = (
        program.reduce_idx == 1
        and program.transform_idx != 5
        and cfg.max_length > 4
    )
    if product_risk:
        report.claims.append(
            VerificationClaim(
                name="product_stability",
                verdict=False,
                risk_level=RISK_WARN,
                message=(
                    "reduce=* without log-domain transform on arrays longer "
                    "than 4 is numerically unstable for |values|>1. Use "
                    "transform=log|x| + post_scale=exp(acc) for stable "
                    "product-magnitude recovery."
                ),
                evidence={
                    "transform": _ELEM_TRANSFORMS[program.transform_idx],
                    "reduce": _REDUCE_OPS[program.reduce_idx],
                    "max_length": cfg.max_length,
                },
            )
        )

    if program.post_scale_idx == 2:
        # exp(acc) — we clamp acc to [-30, 30] so overflow is impossible.
        report.claims.append(
            VerificationClaim(
                name="exp_clamp",
                verdict=True,
                risk_level=RISK_SAFE,
                message=(
                    "post_scale=exp(acc) clamps acc to [-30, 30] before "
                    "exponentiation, bounding output to [0, e^30 ≈ 1e13]."
                ),
            )
        )

    # --- Aggregate ---
    worst_seen = RISK_SAFE
    rank = {RISK_SAFE: 0, RISK_WARN: 1, RISK_HIGH: 2}
    for claim in report.claims:
        if rank[claim.risk_level] > rank[worst_seen]:
            worst_seen = claim.risk_level
    report.worst_risk = worst_seen
    report.overall_safe = all(claim.verdict for claim in report.claims)
    return report


def verify_library(
    entries: list[LibraryEntry],
    config: Optional[VerifierConfig] = None,
) -> list[dict[str, Any]]:
    """Verify every library entry and return a list of `VerificationReport` dicts."""
    reports: list[dict[str, Any]] = []
    for entry in entries:
        report = verify_program(entry.program, config=config)
        report_dict = report.to_dict()
        report_dict["task_name"] = entry.task_name
        report_dict["hit_count"] = int(entry.hit_count)
        reports.append(report_dict)
    return reports


__all__ = [
    "RISK_SAFE",
    "RISK_WARN",
    "RISK_HIGH",
    "RangeBound",
    "VerificationClaim",
    "VerificationReport",
    "VerifierConfig",
    "verify_program",
    "verify_library",
]
