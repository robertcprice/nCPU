"""Pure-Python mirror of the canonical NPCoT executor.

Ported EXACTLY from ``kernels/npcot_wasm/src/lib.rs`` (which is itself the
canonical twin of ``ncpu_metal::npcot_exec``). Every op table, default
branch, and stage order below matches the Rust source line for line:

v1 ``DiscreteProgram`` pipeline::

    acc = init_value(init_idx)
    for x in array[:min(length, len(array))]:
        acc = reduce(acc, transform(x))
    post-scale (none | /max(length, 1) | exp(clamp(acc, -30, 30)))
    + offset

v2 ``ProgramV2`` adds, in front of the v1 pipeline, per data point
(``arity`` floats laid out contiguously)::

    v = combine(fields)          # field select / sum / prod / diff / ...
    if not guard(v): skip        # guard-excluded points are not aggregated
    acc = reduce(acc, transform(v))

and the mean post-scale divides by the number of guard-INCLUDED points,
not the raw length.

Float semantics: the Rust executor uses f32, this mirror uses Python
floats (f64). The registry acceptance tolerance (1e-3 relative to
max(1, max|target|)) absorbs the difference by design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Canonical constants — must match kernels/npcot_wasm/src/lib.rs.
NEG_LARGE = -20.0
LOG_EPS = 1e-6
MAX_ARITY = 4

# Registry acceptance tolerance: max abs error <= TOLERANCE * max(1, max|target|).
TOLERANCE = 1e-3


@dataclass(frozen=True)
class DiscreteProgram:
    """v1 program: init / transform / reduce / post-scale / offset."""

    init_idx: int
    transform_idx: int
    reduce_idx: int
    post_scale_idx: int
    offset: float


@dataclass(frozen=True)
class ProgramV2:
    """v2 program: multi-field records (arity) + combine + predicate guard."""

    arity: int
    combine_idx: int
    guard_idx: int
    guard_threshold: float
    init_idx: int
    transform_idx: int
    reduce_idx: int
    post_scale_idx: int
    offset: float

    @staticmethod
    def from_v1(p: DiscreteProgram) -> "ProgramV2":
        """Exact lift: a v1 program is arity=1, combine=field0, guard=always."""
        return ProgramV2(
            arity=1,
            combine_idx=0,
            guard_idx=0,
            guard_threshold=0.0,
            init_idx=p.init_idx,
            transform_idx=p.transform_idx,
            reduce_idx=p.reduce_idx,
            post_scale_idx=p.post_scale_idx,
            offset=p.offset,
        )

    def is_v1(self) -> bool:
        return self.arity == 1 and self.combine_idx == 0 and self.guard_idx == 0


Program = Union[DiscreteProgram, ProgramV2]


def init_value(init_idx: int) -> float:
    if init_idx == 0:
        return 0.0
    if init_idx == 1:
        return 1.0
    if init_idx == 2:
        return NEG_LARGE
    return 0.0  # Rust: `_ => 0.0`


def apply_transform(x: float, idx: int) -> float:
    if idx == 0:
        return x
    if idx == 1:
        return x * x
    if idx == 2:
        return abs(x)
    if idx == 3:
        return 1.0
    if idx == 4:
        return 1.0 if x > 0.0 else 0.0
    if idx == 5:
        return math.log(abs(x) + LOG_EPS)
    return x  # Rust: `_ => x`


def apply_reduce(acc: float, f: float, idx: int) -> float:
    if idx == 0:
        return acc + f
    if idx == 1:
        return acc * f
    if idx == 2:
        return max(acc, f)
    if idx == 3:
        return min(acc, f)
    return acc + f  # Rust: `_ => acc + f`


def execute_program(program: DiscreteProgram, array: Sequence[float], length: int) -> float:
    """Mirror of Rust ``execute_program``: v1 semantics.

    Note the mean post-scale divides by the raw ``length`` argument
    (clamped to >= 1), NOT the effective iterated length — exactly as
    the Rust source does.
    """
    acc = init_value(program.init_idx)
    effective_len = min(int(length), len(array))
    for i in range(effective_len):
        f_i = apply_transform(array[i], program.transform_idx)
        acc = apply_reduce(acc, f_i, program.reduce_idx)
    if program.post_scale_idx == 0:
        post = acc
    elif program.post_scale_idx == 1:
        post = acc / max(float(length), 1.0)
    else:  # Rust: `_ =>` exp(clamp)
        post = math.exp(max(-30.0, min(30.0, acc)))
    return post + program.offset


def apply_combine(fields: Sequence[float], idx: int) -> float:
    if idx == 1:
        return fields[1] if len(fields) > 1 else 0.0
    if idx == 2:
        return sum(fields)
    if idx == 3:
        prod = 1.0
        for v in fields:
            prod *= v
        return prod
    if idx == 4:
        f0 = fields[0] if fields else 0.0
        f1 = fields[1] if len(fields) > 1 else 0.0
        return f0 - f1
    if idx == 5:
        f0 = fields[0] if fields else 0.0
        f1 = fields[1] if len(fields) > 1 else 0.0
        return abs(f0 - f1)
    if idx == 6:
        return min(fields, default=math.inf)
    if idx == 7:
        return max(fields, default=-math.inf)
    return fields[0] if fields else 0.0  # Rust: `_ => f0`


def guard_passes(v: float, idx: int, t: float) -> bool:
    if idx == 1:
        return v > t
    if idx == 2:
        return v < t
    if idx == 3:
        return abs(v) > t
    if idx == 4:
        return abs(v - t) < 1e-4
    return True  # Rust: `_ => true` (guard 0 = always)


def execute_program_v2(p: ProgramV2, data: Sequence[float], n_points: int) -> float:
    """Mirror of Rust ``execute_program_v2``.

    ``data`` holds ``n_points`` records of ``arity`` floats laid out
    contiguously. The mean post-scale divides by the number of
    guard-INCLUDED points (mean of what was aggregated), not raw length.
    """
    arity = max(int(p.arity), 1)
    usable = min(int(n_points), len(data) // arity)
    acc = init_value(p.init_idx)
    included = 0
    for i in range(usable):
        fields = data[i * arity : (i + 1) * arity]
        v = apply_combine(fields, p.combine_idx)
        if not guard_passes(v, p.guard_idx, p.guard_threshold):
            continue
        included += 1
        acc = apply_reduce(acc, apply_transform(v, p.transform_idx), p.reduce_idx)
    if p.post_scale_idx == 0:
        post = acc
    elif p.post_scale_idx == 1:
        post = acc / max(float(included), 1.0)
    else:  # Rust: `_ =>` exp(clamp)
        post = math.exp(max(-30.0, min(30.0, acc)))
    return post + p.offset


def execute(program: Program, data: Sequence[float], n_points: int) -> float:
    """Dispatch on program version. ``n_points`` is v1's ``length`` /
    v2's record count."""
    if isinstance(program, ProgramV2):
        return execute_program_v2(program, data, n_points)
    return execute_program(program, data, n_points)


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    max_err: float
    first_failure: Optional[Dict[str, Any]]  # {example_index, expected, got}


def verify_program(
    program: Program,
    examples: Sequence[Dict[str, Any]],
    tol: float = TOLERANCE,
) -> VerifyResult:
    """Re-execute ``program`` against every example.

    Each example is ``{"data": [floats...], "n_points": int, "target": float}``.
    Accept iff max abs error <= tol * max(1, max|target|) — the same
    acceptance rule the canonical Rust synthesizer uses, so a program the
    browser tier discovered and verified is accepted here verbatim.
    """
    if not examples:
        return VerifyResult(False, math.inf, {"example_index": -1, "expected": None, "got": None})
    target_scale = 1.0
    for ex in examples:
        target_scale = max(target_scale, abs(float(ex["target"])))
    accept = tol * target_scale

    max_err = 0.0
    first_failure: Optional[Dict[str, Any]] = None
    for i, ex in enumerate(examples):
        expected = float(ex["target"])
        try:
            got: Optional[float] = execute(program, ex["data"], ex["n_points"])
            err = abs(got - expected)
        except (OverflowError, ValueError, ZeroDivisionError):
            got = None
            err = math.inf
        if math.isnan(err):
            err = math.inf
        max_err = max(max_err, err)
        if err > accept and first_failure is None:
            first_failure = {"example_index": i, "expected": expected, "got": got}
    return VerifyResult(first_failure is None, max_err, first_failure)


# ---------------------------------------------------------------------------
# Serialization helpers shared with the registry server.
# ---------------------------------------------------------------------------

V1_FIELDS = ("init_idx", "transform_idx", "reduce_idx", "post_scale_idx", "offset")
V2_FIELDS = (
    "arity",
    "combine_idx",
    "guard_idx",
    "guard_threshold",
    "init_idx",
    "transform_idx",
    "reduce_idx",
    "post_scale_idx",
    "offset",
)
_INT_FIELDS = {
    "arity",
    "combine_idx",
    "guard_idx",
    "init_idx",
    "transform_idx",
    "reduce_idx",
    "post_scale_idx",
}


def program_from_dict(payload: Dict[str, Any], version: int) -> Program:
    """Build a program from its JSON dict. Raises ``ValueError`` on
    missing/malformed fields (every field is required, mirroring the
    Rust loader's ``parse_entry``)."""
    fields = V1_FIELDS if version == 1 else V2_FIELDS
    kwargs: Dict[str, Any] = {}
    for name in fields:
        if name not in payload:
            raise ValueError(f"program missing field: {name}")
        raw = payload[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"program field {name} must be a number")
        kwargs[name] = int(raw) if name in _INT_FIELDS else float(raw)
        if name in _INT_FIELDS and float(raw) != kwargs[name]:
            raise ValueError(f"program field {name} must be an integer")
    if version == 1:
        return DiscreteProgram(**kwargs)
    # Mirror the Rust loader: arity is clamped into 1..=MAX_ARITY.
    kwargs["arity"] = max(1, min(MAX_ARITY, kwargs["arity"]))
    return ProgramV2(**kwargs)


def program_to_dict(program: Program) -> Dict[str, Any]:
    fields = V2_FIELDS if isinstance(program, ProgramV2) else V1_FIELDS
    return {name: getattr(program, name) for name in fields}
