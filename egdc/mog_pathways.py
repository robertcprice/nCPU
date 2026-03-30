"""Persistent pathway and failure memory for Mog synthesis.

This is the beginning of the 'computer built into a model' self-improvement loop:
- successful pathways are stored and counted
- failures are remembered with error types/messages
- family scores evolve from use
- later routing can bias toward historically successful families
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import re
from typing import Any


@dataclass
class SuccessRecord:
    ts: float
    problem_name: str
    family: str
    code: str
    metadata: dict[str, Any]


@dataclass
class FailureRecord:
    ts: float
    problem_name: str
    family: str
    error_type: str
    error_message: str
    metadata: dict[str, Any]


class PathwayMemory:
    def __init__(self, root: str | Path = "egdc/pathway_memory"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.success_file = self.root / "successes.json"
        self.failure_file = self.root / "failures.json"
        self.successes: list[SuccessRecord] = []
        self.failures: list[FailureRecord] = []
        if self.success_file.exists() or self.failure_file.exists():
            self.load()

    def load(self):
        if self.success_file.exists():
            self.successes = [SuccessRecord(**x) for x in json.loads(self.success_file.read_text())]
        if self.failure_file.exists():
            self.failures = [FailureRecord(**x) for x in json.loads(self.failure_file.read_text())]

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.success_file.write_text(json.dumps([asdict(x) for x in self.successes], indent=2))
        self.failure_file.write_text(json.dumps([asdict(x) for x in self.failures], indent=2))

    def record_success(self, problem_name: str, family: str, code: str, metadata: dict[str, Any] | None = None):
        self.successes.append(SuccessRecord(time.time(), problem_name, family, code, metadata or {}))

    def record_failure(self, problem_name: str, family: str, error_type: str, error_message: str, metadata: dict[str, Any] | None = None):
        self.failures.append(FailureRecord(time.time(), problem_name, family, error_type, error_message, metadata or {}))

    def success_count(self, family: str) -> int:
        return sum(1 for s in self.successes if s.family == family)

    def failure_count(self, family: str) -> int:
        return sum(1 for f in self.failures if f.family == family)

    def total_successes(self) -> int:
        return len(self.successes)

    def successes_by_family(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.successes:
            out[s.family] = out.get(s.family, 0) + 1
        return out

    def family_score(self, family: str) -> float:
        s = self.success_count(family)
        f = self.failure_count(family)
        if s == 0 and f == 0:
            return 0.5
        return (s + 1.0) / (s + f + 2.0)

    def known_bad_patterns(self, family: str) -> list[str]:
        return [f.error_message for f in self.failures if f.family == family][-20:]

    def anti_patterns(self, family: str) -> list[str]:
        pats: list[str] = []
        for f in self.failures:
            if f.family != family:
                continue
            ap = f.metadata.get("anti_pattern")
            if ap and ap not in pats:
                pats.append(ap)
        return pats

    def _tokenize_text(self, text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower()))

    def retrieve_similar(self, description: str, signature: str, top_k: int = 5) -> list[dict[str, Any]]:
        query = self._tokenize_text(description + "\n" + signature)
        scored = []
        for s in self.successes:
            basis = str(s.metadata.get("description", "")) + "\n" + str(s.metadata.get("signature", ""))
            toks = self._tokenize_text(basis)
            if not toks:
                sim = 0.0
            else:
                sim = len(query & toks) / len(query | toks)
            scored.append({
                "problem_name": s.problem_name,
                "family": s.family,
                "code": s.code,
                "metadata": s.metadata,
                "similarity": sim,
                "family_score": self.family_score(s.family),
                "score": sim * 0.7 + self.family_score(s.family) * 0.3,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
