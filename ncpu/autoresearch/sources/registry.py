"""Work-item sources for the autoresearch driver.

Each source converts a stream of failed-verification events from a
verifier-gated service into the canonical :class:`WorkItem` shape so the
autoresearch driver can run the cascade on it.

The first concrete source is the verified-skill registry. The registry
re-executes every submitted program against the author's examples; when
that verification fails it returns a 422 with the concrete counterexample.
Those rejected submissions are exactly the shape the cascade wants: a name
plus I/O examples that a cheaper tier failed to satisfy. Mining them into
the driver queue lets the cascade try to *learn the program* (recover
its own verified Mog) and re-POST it to the registry, closing the loop.
"""

from __future__ import annotations

import json
import numbers
from pathlib import Path
from typing import Iterable

from ncpu.autoresearch.types import IoPair, WorkItem


def _example_to_io_pair(example: dict) -> IoPair | None:
    """Translate a registry example ``{data, n_points, target/targets}`` to
    an :class:`IoPair`. The v1/v2 cases use scalar ``target``; v3 uses a
    list ``targets``. v3 trace examples are out of scope for the
    per-shot cascade (the driver runs one call, not a sequence) so they
    are returned as ``None``."""
    if not isinstance(example, dict):
        return None
    if "data" not in example or "n_points" not in example:
        return None
    expected = example.get("target")
    if expected is None:
        # v3 trace example: not representable as a single I/O pair.
        return None
    if isinstance(expected, bool) or not isinstance(expected, numbers.Real):
        return None
    # The cascade solves over integers; accept integers or float-typed
    # integer values (6.0 -> 6) and reject anything else.
    as_int = int(expected)
    if as_int != expected:
        return None
    return IoPair(
        args=[list(example.get("data", []))],
        kwargs={},
        expected=as_int,
    )


def _harness(entry_point: str, pairs: list[IoPair]) -> str:
    """Build a `def check(candidate):` harness from the extracted pairs."""
    lines = ["def check(candidate):"]
    for pair in pairs:
        args_repr = ", ".join(repr(a) for a in pair.args)
        if pair.kwargs:
            kwargs_repr = ", ".join(f"{k}={v!r}" for k, v in pair.kwargs.items())
            call_repr = f"candidate({args_repr}, {kwargs_repr})"
        else:
            call_repr = f"candidate({args_repr})"
        lines.append(f"    assert {call_repr} == {pair.expected!r}")
    if len(lines) == 1:
        lines.append("    pass")
    return "\n".join(lines) + "\n"


def _entry_point_name(skill_name: str) -> str:
    cleaned = "".join(c if (c.isalnum() or c == "_") else "_" for c in skill_name.strip())
    if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"f_{cleaned}" if cleaned else "synthesized"
    return cleaned


def work_item_from_miss(miss: dict, *, task_id: str | None = None) -> WorkItem | None:
    """Translate a single registry miss dict into a :class:`WorkItem`.

    Returns ``None`` for entries that cannot be turned into a per-call
    I/O pair (e.g. v3 trace examples, missing fields)."""
    if not isinstance(miss, dict):
        return None
    name = miss.get("name")
    examples = miss.get("examples")
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(examples, list) or not examples:
        return None
    pairs: list[IoPair] = []
    for raw in examples:
        pair = _example_to_io_pair(raw)
        if pair is not None:
            pairs.append(pair)
    if not pairs:
        return None

    entry_point = _entry_point_name(name)
    return WorkItem(
        task_id=task_id or f"registry/{name}",
        source_benchmark="registry",
        prompt=f"def {entry_point}(*args, **kwargs):\n    \"\"\"Recovered from a rejected registry submission.\"\"\"\n",
        entry_point=entry_point,
        test_source=_harness(entry_point, pairs),
        io_pairs=pairs,
        priority=1.0 + 0.1 * len(pairs),
        provenance={
            "registry_author": miss.get("author"),
            "registry_error": miss.get("error"),
            "registry_first_failure": miss.get("first_failure"),
            "raw_miss": miss,
        },
    )


def mine_registry_misses(
    misses_path: Path,
    out_path: Path,
) -> dict[str, int]:
    """Convert every line of a JSONL misses file into a :class:`WorkItem`.

    Returns counters: ``read``, ``emitted``, ``skipped``. Each emitted
    item is written as a single JSON line to ``out_path``."""
    counters = {"read": 0, "emitted": 0, "skipped": 0}
    if not misses_path.is_file():
        return counters
    # Buffer the emitted rows in memory and only create the output file when
    # at least one item is emitted, so a "skipped everything" run leaves no
    # stray empty queue file behind.
    emitted_lines: list[str] = []
    with misses_path.open() as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            counters["read"] += 1
            try:
                miss = json.loads(line)
            except json.JSONDecodeError:
                counters["skipped"] += 1
                continue
            item = work_item_from_miss(miss)
            if item is None:
                counters["skipped"] += 1
                continue
            emitted_lines.append(json.dumps(item.to_dict()) + "\n")
            counters["emitted"] += 1
    if emitted_lines:
        with out_path.open("w") as dst:
            dst.writelines(emitted_lines)
    return counters


__all__ = [
    "work_item_from_miss",
    "mine_registry_misses",
]
