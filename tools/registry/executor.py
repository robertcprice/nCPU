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


@dataclass(frozen=True)
class ProgramV3:
    """v3 program: persistent state + resets + output select."""

    arity: int
    combine_idx: int
    guard_idx: int
    guard_threshold: float
    reset_guard_idx: int
    reset_threshold: float
    state_init_idx: int
    update_transform_idx: int
    update_reduce_idx: int
    post_scale_idx: int
    output_idx: int
    offset: float

    @staticmethod
    def from_v2(p: ProgramV2) -> "ProgramV3":
        return ProgramV3(
            arity=p.arity,
            combine_idx=p.combine_idx,
            guard_idx=p.guard_idx,
            guard_threshold=p.guard_threshold,
            reset_guard_idx=0,
            reset_threshold=0.0,
            state_init_idx=p.init_idx,
            update_transform_idx=p.transform_idx,
            update_reduce_idx=p.reduce_idx,
            post_scale_idx=p.post_scale_idx,
            output_idx=0,
            offset=p.offset,
        )

    @staticmethod
    def from_v1(p: DiscreteProgram) -> "ProgramV3":
        return ProgramV3.from_v2(ProgramV2.from_v1(p))

    def is_v2(self) -> bool:
        return self.reset_guard_idx == 0 and self.output_idx == 0

    def to_v2(self) -> Optional[ProgramV2]:
        if not self.is_v2():
            return None
        return ProgramV2(
            arity=self.arity,
            combine_idx=self.combine_idx,
            guard_idx=self.guard_idx,
            guard_threshold=self.guard_threshold,
            init_idx=self.state_init_idx,
            transform_idx=self.update_transform_idx,
            reduce_idx=self.update_reduce_idx,
            post_scale_idx=self.post_scale_idx,
            offset=self.offset,
        )


Program = Union[DiscreteProgram, ProgramV2, ProgramV3]


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


def reset_fires(v: float, idx: int, t: float) -> bool:
    return idx != 0 and guard_passes(v, idx, t)


def output_select_v3(p: ProgramV3, s: float, v: float) -> float:
    if p.output_idx == 1:
        return v
    if p.output_idx == 2:
        return s + v
    if p.output_idx == 3:
        return s * v
    if p.output_idx == 4:
        return abs(s)
    return s


def post_scale_v3(p: ProgramV3, y: float, included: int) -> float:
    if p.post_scale_idx == 0:
        return y
    if p.post_scale_idx == 1:
        return y / max(float(included), 1.0)
    return math.exp(max(-30.0, min(30.0, y)))


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


def execute_program_v3(p: ProgramV3, data: Sequence[float], n_steps: int) -> List[float]:
    """Mirror of Rust ``execute_program_v3``.

    ``data`` holds ``n_steps`` records of ``arity`` floats laid out
    contiguously. The inclusion guard controls whether state is updated;
    the reset guard restores state and the included counter before each
    included update.
    """
    arity = max(int(p.arity), 1)
    usable = min(int(n_steps), len(data) // arity)
    s = init_value(p.state_init_idx)
    included = 0
    outputs: List[float] = []
    for i in range(usable):
        fields = data[i * arity : (i + 1) * arity]
        v = apply_combine(fields, p.combine_idx)
        if guard_passes(v, p.guard_idx, p.guard_threshold):
            if reset_fires(v, p.reset_guard_idx, p.reset_threshold):
                s = init_value(p.state_init_idx)
                included = 0
            included += 1
            s = apply_reduce(s, apply_transform(v, p.update_transform_idx), p.update_reduce_idx)
        y = output_select_v3(p, s, v)
        outputs.append(post_scale_v3(p, y, included) + p.offset)
    return outputs


def execute_program_v3_final(p: ProgramV3, data: Sequence[float], n_steps: int) -> float:
    """Final-step output of a v3 replay, including empty-trace semantics."""
    outputs = execute_program_v3(p, data, n_steps)
    if outputs:
        return outputs[-1]
    s = init_value(p.state_init_idx)
    y = output_select_v3(p, s, 0.0)
    return post_scale_v3(p, y, 0) + p.offset


def execute(program: Program, data: Sequence[float], n_points: int) -> float:
    """Dispatch on program version. ``n_points`` is v1's ``length`` /
    v2's record count / v3's step count."""
    if isinstance(program, ProgramV3):
        return execute_program_v3_final(program, data, n_points)
    if isinstance(program, ProgramV2):
        return execute_program_v2(program, data, n_points)
    return execute_program(program, data, n_points)


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    max_err: float
    first_failure: Optional[Dict[str, Any]]  # {example_index, expected, got}


def _example_targets(ex: Dict[str, Any]) -> List[float]:
    if "targets" in ex:
        targets = ex["targets"]
        if not isinstance(targets, list) or not targets:
            raise ValueError("example targets must be a non-empty list")
        return [float(t) for t in targets]
    return [float(ex["target"])]


def verify_program(
    program: Program,
    examples: Sequence[Dict[str, Any]],
    tol: float = TOLERANCE,
) -> VerifyResult:
    """Re-execute ``program`` against every example.

    v1/v2 examples are ``{"data": [floats...], "n_points": int, "target": float}``.
    v3 trace examples may use ``"targets": [step outputs...]`` to verify the
    full replay; otherwise the final-step output is compared to ``"target"``.
    Accept iff max abs error <= tol * max(1, max|target|) — the same
    acceptance rule the canonical Rust synthesizer uses, so a program the
    browser tier discovered and verified is accepted here verbatim.
    """
    if not examples:
        return VerifyResult(False, math.inf, {"example_index": -1, "expected": None, "got": None})
    try:
        target_values = [_target for ex in examples for _target in _example_targets(ex)]
    except (KeyError, TypeError, ValueError):
        return VerifyResult(False, math.inf, {"example_index": -1, "expected": None, "got": None})
    target_scale = max([1.0] + [abs(t) for t in target_values])
    accept = tol * target_scale

    max_err = 0.0
    first_failure: Optional[Dict[str, Any]] = None
    for i, ex in enumerate(examples):
        expected_values: Optional[List[float]] = None
        try:
            expected_values = _example_targets(ex)
            if isinstance(program, ProgramV3) and "targets" in ex:
                if len(expected_values) != int(ex["n_points"]):
                    raise ValueError("v3 targets length must match n_points")
                got_values = execute_program_v3(program, ex["data"], ex["n_points"])
                for step, (expected, got) in enumerate(zip(expected_values, got_values)):
                    err = abs(got - expected)
                    if math.isnan(err):
                        err = math.inf
                    max_err = max(max_err, err)
                    if err > accept and first_failure is None:
                        first_failure = {
                            "example_index": i,
                            "step": step,
                            "expected": expected,
                            "got": got,
                        }
                if len(got_values) < len(expected_values) and first_failure is None:
                    first_failure = {
                        "example_index": i,
                        "step": len(got_values),
                        "expected": expected_values[len(got_values)],
                        "got": None,
                    }
                continue
            expected = expected_values[0]
            got: Optional[float] = execute(program, ex["data"], ex["n_points"])
            err = abs(got - expected)
        except (OverflowError, ValueError, ZeroDivisionError, IndexError):
            got = None
            err = math.inf
        if math.isnan(err):
            err = math.inf
        max_err = max(max_err, err)
        if err > accept and first_failure is None:
            expected_for_failure = expected_values[0] if expected_values else None
            first_failure = {"example_index": i, "expected": expected_for_failure, "got": got}
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
V3_FIELDS = (
    "arity",
    "combine_idx",
    "guard_idx",
    "guard_threshold",
    "reset_guard_idx",
    "reset_threshold",
    "state_init_idx",
    "update_transform_idx",
    "update_reduce_idx",
    "post_scale_idx",
    "output_idx",
    "offset",
)
_INT_FIELDS = {
    "arity",
    "combine_idx",
    "guard_idx",
    "reset_guard_idx",
    "state_init_idx",
    "update_transform_idx",
    "update_reduce_idx",
    "post_scale_idx",
    "output_idx",
}


def program_from_dict(payload: Dict[str, Any], version: int) -> Program:
    """Build a program from its JSON dict. Raises ``ValueError`` on
    missing/malformed fields (every field is required, mirroring the
    Rust loader's ``parse_entry``)."""
    if version == 1:
        fields = V1_FIELDS
    elif version == 2:
        fields = V2_FIELDS
    elif version == 3:
        fields = V3_FIELDS
    else:
        raise ValueError(f"unsupported program version: {version}")
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
    if version == 2:
        return ProgramV2(**kwargs)
    return ProgramV3(**kwargs)


def program_to_dict(program: Program) -> Dict[str, Any]:
    if isinstance(program, ProgramV3):
        fields = V3_FIELDS
    elif isinstance(program, ProgramV2):
        fields = V2_FIELDS
    else:
        fields = V1_FIELDS
    return {name: getattr(program, name) for name in fields}
