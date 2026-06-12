"""Phase B — natural-language → program corpus and (eventually) encoder.

Rung 9 Phase B: a model that maps an NL prompt directly to a program in a
Turing-complete representation (Mog / register machine) and reasons in that
space, rather than emitting prose tokens. This package currently holds the
training-corpus tooling; the encoder lands here next.
"""

from ncpu.phase_b.corpus import (
    CorpusRecord,
    CorpusStats,
    read_corpus,
    summarize,
    write_corpus,
)

__all__ = [
    "CorpusRecord",
    "CorpusStats",
    "read_corpus",
    "summarize",
    "write_corpus",
]
