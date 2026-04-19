"""Autoresearch daemon — learn from hard-fails, grow the library.

The autoresearch loop closes on the verifier that powers the compounding
safety stack. Every eval run produces a JSON with per-problem pass/fail
records. `miner.py` reads those records, extracts the failing problems
along with their test suites, and emits a prioritized work queue. The
`cascade.py` solver pulls from the queue and tries to find a program that
passes the tests using escalating techniques (cached library → nsynth
enumerative → nsynth full → LLM teacher). Each successful solve is
`distiller.py`'d into a persistent artifact the library can grow from.

The whole thing is budget-aware: wall seconds, USD cost, max problems per
session. It is safe to interrupt at any time; partial progress is
checkpointed to JSONL files so the next run resumes.
"""

from ncpu.autoresearch.types import (
    Budget,
    IoPair,
    SolvedItem,
    WorkItem,
)

__all__ = [
    "Budget",
    "IoPair",
    "SolvedItem",
    "WorkItem",
]
