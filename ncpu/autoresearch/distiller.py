"""Persist solved items + grow the library.

Two outputs:

1. ``solved_programs.jsonl`` — an always-on append-only log keyed by
   ``task_id``. Every successful cascade run writes one row containing the
   Python source, the solver that produced it, wall time, and
   provenance. This is the dataset for any downstream paper table /
   library-rebuild / LLM-teacher distillation pipeline.

2. A live :class:`ArrayProgramLibrary` update — when a solved item is
   translatable into a ``DiscreteArrayProgram`` (array-reduction shape
   with integer I/O), the entry is written straight into the library
   JSON. This is how the autoresearch loop produces in-place library
   growth visible at the next eval.

The library-update path is best-effort: items that can't be translated
(string manipulation, nested control flow, etc.) remain in the JSONL
and are available for manual review / teacher-distillation / paper.

5-tuple translation (:func:`translate_to_5tuple`) is pure Python — it
mirrors the ``DiscreteArrayProgram`` executor semantics without torch,
so the offline ``cli distill`` path can run on any host. Library
entries need a hidden-state *signature* to fire at inference;
``ArrayProgramLibrary.load`` crashes on ``signature: null`` entries
(``LibraryEntry.from_dict`` iterates the signature list), and a
fabricated signature would fire spuriously at inference. Offline
distillation therefore parks translatable programs in a sidecar
``pending_distill.json`` next to the library; the *driver* path — which
has a live model and can capture real hidden states — records keyed
entries straight into the library JSON.
"""

from __future__ import annotations

import ast
import builtins
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from ncpu.autoresearch.types import IoPair, SolvedItem, WorkItem


def append_solved(solved: SolvedItem, *, out_path: Path) -> None:
    """Append one SolvedItem to solved_programs.jsonl."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as fh:
        fh.write(json.dumps(solved.to_dict()) + "\n")


def load_solved(path: Path) -> list[SolvedItem]:
    """Load all SolvedItems from the persistent JSONL."""
    items: list[SolvedItem] = []
    if not path.exists():
        return items
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(SolvedItem(**{k: v for k, v in d.items() if k in SolvedItem.__dataclass_fields__}))
    return items


def dedupe_solved(path: Path) -> int:
    """Keep the latest SolvedItem per task_id. Returns count after dedupe."""
    items = load_solved(path)
    latest: dict[str, SolvedItem] = {}
    for it in items:
        latest[it.task_id] = it
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for it in latest.values():
            fh.write(json.dumps(it.to_dict()) + "\n")
    return len(latest)


def summarize_solved(items: Iterable[SolvedItem]) -> dict:
    """Counts by solver + total wall time."""
    by_solver: dict[str, int] = {}
    total_wall = 0.0
    for it in items:
        by_solver[it.solver] = by_solver.get(it.solver, 0) + 1
        total_wall += it.wall_seconds
    return {
        "total_solved": sum(by_solver.values()),
        "by_solver": by_solver,
        "total_wall_seconds": round(total_wall, 2),
    }


# ---------------------------------------------------------------------------
# Pure-Python mirror of the DiscreteArrayProgram executor
# ---------------------------------------------------------------------------
# The canonical executor lives in
# ncpu.self_optimizing.array_program_library.DiscreteArrayProgram.execute and
# is torch-based. The constants below mirror the program space defined in
# ncpu.self_optimizing.array_executable_thought_head (init choices, element
# transforms, reduce ops, post scales) so this module stays importable
# without torch. test_distiller_library.py cross-checks both the constants
# and the executor semantics against the torch implementation.

_PURE_INIT_VALUES: tuple[float, ...] = (0.0, 1.0, -20.0)  # "0", "1", "-large"
_N_INIT = 3        # len(_INIT_CHOICES)
_N_TRANSFORM = 6   # len(_ELEM_TRANSFORMS): x, x*x, |x|, 1, 1{x>0}, log|x|
_N_REDUCE = 4      # len(_REDUCE_OPS): +, *, max, min
_N_POST_SCALE = 3  # len(_POST_SCALES): acc, acc/len, exp(acc)
_PURE_LOG_EPS = 1e-6
_PURE_EXP_CLAMP = 30.0

PROGRAM_SPACE_SIZE = _N_INIT * _N_TRANSFORM * _N_REDUCE * _N_POST_SCALE
"""Number of discrete program shapes (216); offset is the continuous 5th."""


def _pure_transform(x: float, idx: int) -> float:
    if idx == 0:
        return x
    if idx == 1:
        return x * x
    if idx == 2:
        return abs(x)
    if idx == 3:
        return 1.0
    if idx == 4:
        return 1.0 if x > 0 else 0.0
    if idx == 5:
        return math.log(abs(x) + _PURE_LOG_EPS)
    raise ValueError(f"unknown transform index {idx}")


def _pure_reduce(acc: float, f: float, idx: int) -> float:
    if idx == 0:
        return acc + f
    if idx == 1:
        return acc * f
    if idx == 2:
        return max(acc, f)
    if idx == 3:
        return min(acc, f)
    raise ValueError(f"unknown reduce index {idx}")


def execute_program_pure(
    program: Mapping[str, Any], values: Sequence[float]
) -> float:
    """Pure-Python mirror of ``DiscreteArrayProgram.execute`` (single array).

    ``program`` carries the 5-tuple keys ``init_idx``, ``transform_idx``,
    ``reduce_idx``, ``post_scale_idx``, ``offset`` — the exact payload
    ``DiscreteArrayProgram.from_dict`` accepts. ``values`` is the array
    (already truncated to its true length; no padding/mask semantics).
    """
    init_idx = int(program["init_idx"])
    transform_idx = int(program["transform_idx"])
    reduce_idx = int(program["reduce_idx"])
    post_scale_idx = int(program["post_scale_idx"])
    offset = float(program["offset"])
    if not 0 <= init_idx < _N_INIT:
        raise ValueError(f"init_idx out of range: {init_idx}")
    if not 0 <= transform_idx < _N_TRANSFORM:
        raise ValueError(f"transform_idx out of range: {transform_idx}")
    if not 0 <= reduce_idx < _N_REDUCE:
        raise ValueError(f"reduce_idx out of range: {reduce_idx}")
    if not 0 <= post_scale_idx < _N_POST_SCALE:
        raise ValueError(f"post_scale_idx out of range: {post_scale_idx}")

    acc = _PURE_INIT_VALUES[init_idx]
    for raw in values:
        x = float(raw)
        acc = _pure_reduce(acc, _pure_transform(x, transform_idx), reduce_idx)

    if post_scale_idx == 0:
        post = acc
    elif post_scale_idx == 1:
        post = acc / max(float(len(values)), 1.0)
    else:
        post = math.exp(min(max(acc, -_PURE_EXP_CLAMP), _PURE_EXP_CLAMP))
    return post + offset


# ---------------------------------------------------------------------------
# 5-tuple translation: verified Python solve -> DiscreteArrayProgram dict
# ---------------------------------------------------------------------------

# Deterministic behavior-fit probe set: 30 fixed int arrays covering
# single-element, negatives, zeros, duplicates, sorted/unsorted, and
# mixed-sign inputs. Lengths stay <= 8 to mirror the deployed
# array_max_len; magnitudes stay small so float arithmetic is exact.
_PROBE_ARRAYS: tuple[tuple[int, ...], ...] = (
    (0,), (1,), (-1,), (7,), (-9,),
    (0, 0), (1, 1), (-1, 1), (2, 3), (-2, -3),
    (5, -5), (0, 4), (3, 0), (-7, 0),
    (1, 2, 3), (3, 2, 1), (-1, -2, -3), (2, -3, 5),
    (0, 0, 0), (4, 4, 4), (1, -1, 1, -1),
    (1, 2, 3, 4, 5), (5, 4, 3, 2, 1), (-5, 3, -2, 8, 0),
    (2, 2, 3, 3, 1), (9, -9, 8, -8, 7, -7),
    (1, 0, -1, 0, 1, 0, -1), (6, 1, 5, 2, 4, 3, 7, 0),
    (-4, -4, -4, -4, -4, -4, -4, -4), (1, 1, 2, 2, 3, 3, 4, 4),
)

_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "callable", "chr", "dict", "divmod",
    "enumerate", "filter", "float", "frozenset", "hash", "int",
    "isinstance", "issubclass", "iter", "len", "list", "map", "max",
    "min", "next", "ord", "pow", "range", "repr", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    "ArithmeticError", "AssertionError", "AttributeError", "Exception",
    "IndexError", "KeyError", "LookupError", "NotImplementedError",
    "OverflowError", "RuntimeError", "StopIteration", "TypeError",
    "ValueError", "ZeroDivisionError",
)

# Solved programs are pipeline-verified, not raw user input — but keep the
# import surface minimal anyway. HumanEval prompts routinely start with
# `from typing import List`, so a small stdlib allowlist is required.
_SAFE_IMPORT_MODULES = frozenset({
    "typing", "math", "itertools", "functools", "collections",
    "heapq", "bisect", "string", "re", "operator",
})


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in _SAFE_IMPORT_MODULES:
        raise ImportError(f"import of {name!r} not allowed in distiller sandbox")
    return __import__(name, globals, locals, fromlist, level)


def _restricted_namespace() -> dict[str, Any]:
    safe = {
        name: getattr(builtins, name)
        for name in _SAFE_BUILTIN_NAMES
        if hasattr(builtins, name)
    }
    safe["__import__"] = _safe_import
    return {"__builtins__": safe}


def io_pair_int_array(pair: IoPair) -> Optional[list[int]]:
    """Return the single int-list argument of ``pair``, or None if it isn't one."""
    if pair.kwargs:
        return None
    if len(pair.args) != 1:
        return None
    arr = pair.args[0]
    if not isinstance(arr, list):
        return None
    for value in arr:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
    return arr


def _is_scalar_number(value: Any) -> bool:
    if not isinstance(value, (bool, int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _load_callable(
    solved: SolvedItem,
    *,
    prompt: Optional[str],
    entry_point: Optional[str],
) -> Optional[Callable[..., Any]]:
    """Exec the solved source in a restricted namespace, return the entry point.

    ``program_python`` is usually a *continuation* of the WorkItem prompt
    (HumanEval-style function body), so try ``prompt + program_python``
    first, then the program alone (full-def style).
    """
    preferred: list[str] = []
    for name in (entry_point, (solved.provenance or {}).get("entry_point")):
        if name and name not in preferred:
            preferred.append(name)
    if prompt:
        match = re.search(r"def\s+(\w+)\s*\(", prompt)
        if match and match.group(1) not in preferred:
            preferred.append(match.group(1))

    sources: list[str] = []
    if prompt:
        sources.append(prompt + solved.program_python)
    sources.append(solved.program_python)

    for source in sources:
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        namespace = _restricted_namespace()
        try:
            exec(compile(tree, "<distill>", "exec"), namespace)  # noqa: S102
        except Exception:  # noqa: BLE001 — refusal, not failure
            continue
        defined = [
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        ]
        for name in preferred + defined[::-1]:
            fn = namespace.get(name)
            if callable(fn):
                return fn
    return None


def _call_behavior(fn: Callable[..., Any], arr: Sequence[int]) -> Optional[Any]:
    """Call ``fn`` on a fresh copy of ``arr``; None on raise / non-scalar."""
    try:
        result = fn(list(arr))
    except Exception:  # noqa: BLE001 — refusal, not failure
        return None
    if not _is_scalar_number(result):
        return None
    return result


def _fit_offset(
    idxs: tuple[int, int, int, int],
    points: Sequence[tuple[list[int], Any]],
) -> Optional[float]:
    """Fit the constant offset for a discrete shape; exact match on all points."""
    init_idx, transform_idx, reduce_idx, post_scale_idx = idxs
    base_program = {
        "init_idx": init_idx,
        "transform_idx": transform_idx,
        "reduce_idx": reduce_idx,
        "post_scale_idx": post_scale_idx,
        "offset": 0.0,
    }
    offset: Optional[float] = None
    for arr, target in points:
        base = execute_program_pure(base_program, arr)
        if not math.isfinite(base):
            return None
        if offset is None:
            try:
                offset = float(target) - base
            except (OverflowError, ValueError):
                return None
            if not math.isfinite(offset):
                return None
            if offset == 0.0:
                offset = 0.0  # normalize -0.0
        # Exact match only — Python compares int == float exactly.
        if base + offset != target:
            return None
    return offset


def translate_to_5tuple(
    solved: SolvedItem,
    io_pairs: list[IoPair],
    *,
    prompt: Optional[str] = None,
    entry_point: Optional[str] = None,
) -> Optional[dict]:
    """Translate a verified Python solve into a ``DiscreteArrayProgram`` dict.

    Only int-array → scalar behaviors are attempted (single list-of-ints
    argument, numeric output) — the shape ``DiscreteArrayProgram`` executes.
    The solved source is exec'd in a restricted namespace; the resulting
    callable is treated as ground truth over the given ``io_pairs`` plus the
    deterministic ``_PROBE_ARRAYS`` set; the 216-shape discrete space is
    enumerated exhaustively (offset fitted per shape) and the first program
    matching the callable *exactly on every probe* is returned as a
    ``DiscreteArrayProgram.from_dict``-compatible dict. Returns None on any
    mismatch — honest refusal, never a lossy approximation.

    Pure Python by design: no torch import, runs anywhere.
    """
    arrays: list[list[int]] = []
    expecteds: list[Any] = []
    for pair in io_pairs:
        arr = io_pair_int_array(pair)
        if arr is None or not _is_scalar_number(pair.expected):
            return None
        arrays.append(arr)
        expecteds.append(pair.expected)

    fn = _load_callable(solved, prompt=prompt, entry_point=entry_point)
    if fn is None:
        return None

    # Sanity gate: the callable must reproduce the recorded expecteds
    # exactly, otherwise we exec'd the wrong entry point.
    for arr, expected in zip(arrays, expecteds):
        got = _call_behavior(fn, arr)
        if got is None or got != expected:
            return None

    points: list[tuple[list[int], Any]] = []
    seen: set[tuple[int, ...]] = set()
    for arr in arrays + [list(p) for p in _PROBE_ARRAYS]:
        key = tuple(arr)
        if key in seen:
            continue
        seen.add(key)
        target = _call_behavior(fn, arr)
        if target is None:
            return None
        points.append((arr, target))

    for init_idx in range(_N_INIT):
        for transform_idx in range(_N_TRANSFORM):
            for reduce_idx in range(_N_REDUCE):
                for post_scale_idx in range(_N_POST_SCALE):
                    idxs = (init_idx, transform_idx, reduce_idx, post_scale_idx)
                    offset = _fit_offset(idxs, points)
                    if offset is not None:
                        return {
                            "init_idx": init_idx,
                            "transform_idx": transform_idx,
                            "reduce_idx": reduce_idx,
                            "post_scale_idx": post_scale_idx,
                            "offset": float(offset),
                        }
    return None


# ---------------------------------------------------------------------------
# Offline distillation pass
# ---------------------------------------------------------------------------

PENDING_DISTILL_NAME = "pending_distill.json"
_PENDING_SCHEMA = "autoresearch.pending_distill/1"


def _load_pending(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
        if isinstance(payload.get("pending"), dict):
            return {"schema": _PENDING_SCHEMA, "pending": payload["pending"]}
    return {"schema": _PENDING_SCHEMA, "pending": {}}


def distill_solved(
    items: Iterable[SolvedItem],
    library_path: Path,
    *,
    work_items: Optional[Mapping[str, WorkItem]] = None,
    pending_path: Optional[Path] = None,
) -> dict:
    """Offline distillation pass: translate solves, park them for the library.

    For each translatable :class:`SolvedItem`, the 5-tuple program is merged
    (keyed by ``task_id``, latest wins) into ``pending_distill.json`` next to
    ``library_path``. The library JSON itself is **never** written here:
    offline distillation has no hidden state, ``ArrayProgramLibrary``
    requires a real signature per entry (``load`` raises on ``signature:
    null`` — see test_distiller_library), and a fabricated signature would
    fire spuriously at inference. The driver path
    (:func:`ncpu.autoresearch.driver.grow_library_from_solve`) is where
    signature-keyed entries are recorded with a live model.

    ``work_items`` (``task_id -> WorkItem``) recovers prompts / entry points
    / io_pairs for body-style programs; without it, only standalone full-def
    programs can be exec'd.

    Returns a JSON-serializable summary report.
    """
    library_path = Path(library_path)
    pending_file = (
        Path(pending_path)
        if pending_path is not None
        else library_path.parent / PENDING_DISTILL_NAME
    )
    lookup = work_items or {}
    payload = _load_pending(pending_file)

    total = 0
    translated_ids: list[str] = []
    refused_ids: list[str] = []
    for item in items:
        total += 1
        wi = lookup.get(item.task_id)
        five = item.program_5tuple
        if five is None:
            five = translate_to_5tuple(
                item,
                wi.io_pairs if wi is not None else [],
                prompt=wi.prompt if wi is not None else None,
                entry_point=(
                    wi.entry_point
                    if wi is not None
                    else (item.provenance or {}).get("entry_point")
                ),
            )
        if five is None:
            refused_ids.append(item.task_id)
            continue
        translated_ids.append(item.task_id)
        payload["pending"][item.task_id] = {
            "task_id": item.task_id,
            "source_benchmark": item.source_benchmark,
            "solver": item.solver,
            "program": five,
            "signature": None,
            "pending_signature": True,
        }

    pending_file.parent.mkdir(parents=True, exist_ok=True)
    pending_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    library_entries: Optional[int] = None
    if library_path.exists():
        try:
            lib_payload = json.loads(library_path.read_text(encoding="utf-8"))
            library_entries = len(lib_payload.get("entries", []))
        except (json.JSONDecodeError, OSError):
            library_entries = None

    return {
        "total_items": total,
        "translated": len(translated_ids),
        "refused": len(refused_ids),
        "translated_task_ids": translated_ids,
        "refused_task_ids": refused_ids,
        "pending_path": str(pending_file),
        "pending_total": len(payload["pending"]),
        "library_path": str(library_path),
        "library_entries": library_entries,
    }
