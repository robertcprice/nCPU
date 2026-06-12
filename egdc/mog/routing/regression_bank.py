"""Auto-growing regression bank for Mog synthesis.

When the orchestrator hits a failure, it can automatically add a regression
entry here. This grows the system's test coverage from real usage, not just
from hand-authored benchmark problems.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from typing import Any


@dataclass
class RegressionEntry:
    ts: float
    problem_name: str
    description: str
    code: str
    error: str
    test_input: str
    expected_output: str
    metadata: dict[str, Any] | None = None


class RegressionBank:
    def __init__(self, root: str | Path = "egdc/regression_bank"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = self.root / "regressions.json"
        self.regressions: list[RegressionEntry] = []
        if self.file.exists():
            self.load()

    def load(self):
        if self.file.exists():
            self.regressions = [RegressionEntry(**x) for x in json.loads(self.file.read_text())]

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps([asdict(x) for x in self.regressions], indent=2))

    def add_regression(
        self,
        problem_name: str,
        description: str,
        code: str,
        error: str,
        test_input: str,
        expected_output: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.regressions.append(RegressionEntry(
            ts=time.time(),
            problem_name=problem_name,
            description=description,
            code=code,
            error=error,
            test_input=test_input,
            expected_output=expected_output,
            metadata=metadata,
        ))

    def has_regression_for(self, problem_name: str) -> bool:
        return any(r.problem_name == problem_name for r in self.regressions)

    def get_regressions(self, limit: int = 100) -> list[RegressionEntry]:
        return self.regressions[-limit:]
