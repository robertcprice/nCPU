"""Synthesis-API refusal source for the autoresearch driver.

A `success: false` response from `ncpu.synthesis_api.server` is exactly the
shape the cascade wants: a name + I/O pairs that a cheaper tier failed
to satisfy. Mine those refusals into the same WorkItem queue as
``ncpu.autoresearch.sources.registry`` so the driver runs them through
the cascade uniformly.

The capture point is the API's refusal branch — see
``docs/autoresearch_continuous.md`` §4. Every entry in the
``refusals.jsonl`` has the shape::

    {
        "name": "junk_scalar",
        "examples": [{"inputs": [1], "expected": 7}, ...],
        "error": "no program found",
        "ts": "2026-06-14T20:00:00+00:00",
    }

The mining rule is intentionally strict: the inputs must be int / [int] /
str, the expected must be int, kwargs are not representable. Anything
else is skipped (and recorded in the counter so the runner can see why
the queue isn't growing).
"""

from __future__ import annotations

import json
import numbers
from pathlib import Path

from ncpu.autoresearch.types import IoPair, WorkItem


def _example_to_io_pair(example: dict) -> IoPair | None:
    if not isinstance(example, dict):
        return None
    inputs = example.get("inputs")
    expected = example.get("expected")
    if not isinstance(inputs, list) or not inputs:
        return None
    if isinstance(expected, bool) or not isinstance(expected, numbers.Integral):
        return None
    # Keyword arguments are not representable in nsynth's positional
    # problem format; skip any example that carries them rather than
    # fabricate a synthetic value.
    if example.get("kwargs"):
        return None
    coerced: list = []
    for v in inputs:
        if isinstance(v, bool) or not isinstance(v, (int, str, list)):
            return None
        if isinstance(v, list) and not all(
            isinstance(x, int) and not isinstance(x, bool) for x in v
        ):
            return None
        coerced.append(v)
    return IoPair(args=coerced, kwargs={}, expected=int(expected))


def _entry_point_name(skill_name: str) -> str:
    cleaned = "".join(c if (c.isalnum() or c == "_") else "_" for c in skill_name.strip())
    if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"f_{cleaned}" if cleaned else "synthesized"
    return cleaned


def work_item_from_refusal(refusal: dict, *, task_id: str | None = None) -> WorkItem | None:
    if not isinstance(refusal, dict):
        return None
    name = refusal.get("name")
    examples = refusal.get("examples")
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(examples, list) or not examples:
        return None
    pairs: list[IoPair] = []
    skipped = 0
    for raw in examples:
        pair = _example_to_io_pair(raw)
        if pair is None:
            skipped += 1
        else:
            pairs.append(pair)
    if not pairs:
        return None
    entry_point = _entry_point_name(name)
    return WorkItem(
        task_id=task_id or f"synthesis_api/{name}",
        source_benchmark="synthesis_api",
        prompt=f"def {entry_point}(*args, **kwargs):\n    \"\"\"Recovered from a synthesis API refusal.\"\"\"\n",
        entry_point=entry_point,
        test_source="def check(candidate):\n    pass\n",
        io_pairs=pairs,
        priority=1.0 + 0.1 * len(pairs),
        provenance={
            "synth_error": refusal.get("error"),
            "synth_method": refusal.get("method"),
            "synth_elapsed_ms": refusal.get("elapsed_ms"),
            "synth_ts": refusal.get("ts"),
            "raw_refusal": refusal,
            "skipped_examples": skipped,
        },
    )


def mine_synthesis_refusals(
    refusals_path: Path,
    out_path: Path,
) -> dict[str, int]:
    counters = {"read": 0, "emitted": 0, "skipped": 0}
    if not refusals_path.is_file():
        return counters
    emitted_lines: list[str] = []
    with refusals_path.open() as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            counters["read"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counters["skipped"] += 1
                continue
            item = work_item_from_refusal(rec)
            if item is None:
                counters["skipped"] += 1
                continue
            emitted_lines.append(json.dumps(item.to_dict()) + "\n")
            counters["emitted"] += 1
    if emitted_lines:
        with out_path.open("w") as dst:
            dst.writelines(emitted_lines)
    return counters


__all__ = ["work_item_from_refusal", "mine_synthesis_refusals"]
