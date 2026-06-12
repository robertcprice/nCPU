"""Phase B training corpus — the (natural-language, program, I/O) schema.

Rung 9 Phase B is a model that maps a natural-language prompt directly to
a *program* (not to tokens of prose), and reasons in that program space.
Training it needs a corpus of aligned triples:

    nl_prompt  →  program  (verified to reproduce)  →  io_pairs

This module defines that record and its JSONL store. It is deliberately
program-language-agnostic: a record may carry a Mog program (from the
differentiable-compiler benchmark) or Python (from autoresearch's verified
HumanEval solves). What every record shares is that the program is
*verified* to reproduce its io_pairs — the corpus never contains an
unchecked (prompt, program) guess, the same honest-by-construction
discipline as the synthesizer itself.

The diagnostic that motivated this: the array→scalar fold space
(DiscreteArrayProgram, 288 programs) cannot express list/tuple/boolean
outputs or stateful scans — which is most real code. So Phase B must
reason in a Turing-complete program representation (Mog / register
machine), and this corpus is built in that space from day one.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class CorpusRecord:
    """One aligned (natural-language → verified program → I/O) training triple."""

    source: str  # "egdc_factory" | "autoresearch_humaneval"
    task_id: str  # "add_two_v0" | "HumanEval/42"
    nl_prompt: str  # the natural-language description / docstring
    entry_point: str  # name of the function the program defines
    signature: str  # "fn add_two(a: i64, b: i64) -> i64" or a python def line
    program_lang: str  # "mog" | "python"
    program_src: str  # the verified program source
    io_pairs: list[dict[str, Any]]  # [{"inputs": [...], "expected": <value|str>}]
    category: str = "uncategorized"
    verified: bool = False  # program reproduces every io_pair

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CorpusRecord":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def write_corpus(records: Iterable[CorpusRecord], out_path: Path) -> int:
    """Write records to a JSONL file. Returns the count written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict()) + "\n")
            n += 1
    return n


def read_corpus(path: Path) -> list[CorpusRecord]:
    """Load all records from a JSONL corpus file."""
    out: list[CorpusRecord] = []
    if not path.exists():
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(CorpusRecord.from_dict(json.loads(line)))
    return out


@dataclass
class CorpusStats:
    total: int = 0
    verified: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    by_lang: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    total_io_pairs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(records: list[CorpusRecord]) -> CorpusStats:
    """Aggregate a corpus into a stats summary."""
    return CorpusStats(
        total=len(records),
        verified=sum(1 for r in records if r.verified),
        by_source=dict(Counter(r.source for r in records)),
        by_lang=dict(Counter(r.program_lang for r in records)),
        by_category=dict(Counter(r.category for r in records)),
        total_io_pairs=sum(len(r.io_pairs) for r in records),
    )
