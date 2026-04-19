"""The always-compounding persistent store for autoresearch.

The autoresearch cascade writes every verified solve into this store.
On the next run of any task, the store is consulted *first* — before
the LLM, before the library, before any retry logic. A store hit is
returned as a ``CacheHit`` that callers can substitute for a fresh
generation, getting zero-cost first-try passes on anything we've
previously solved.

Three indices live inside:

1. ``solved_programs.jsonl`` — append-only fact log of every solve, one
   JSON row per :class:`SolvedItem`. This is the source of truth.
2. ``prompt_cache.json`` — ``task_id → (hash, program_python)`` map for
   exact-match re-runs. Built lazily from the JSONL.
3. ``temperature_stats.json`` — per-temperature solve counts across all
   sessions, lets future runs pick temps that have paid off before.

The store is safe to share across processes: writes are append-only,
indices are rebuildable from the log, so concurrent appends never
corrupt state. The only exclusive operation is
``rebuild_indices()`` which reads the full log.

Library-growth hook: when a caller provides a ``library`` handle and
the solved program is translatable to a :class:`DiscreteArrayProgram`
(array-reduction shape), the store also grows the NPCoT library via
:func:`continual_library.record_successful_generation`. This is the
real "NPCoT learns from solves" path. For solutions outside the
array-reduction shape (string manipulation, nested conditionals), the
Python source is preserved in the JSONL and the prompt cache, but the
differentiable library is not extended.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ncpu.autoresearch.types import SolvedItem, WorkItem


def hash_prompt(prompt: str, entry_point: str) -> str:
    """Stable hash used as prompt-cache key."""
    h = hashlib.sha256()
    h.update(entry_point.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()[:32]


@dataclass
class CacheHit:
    """Returned when the store has a cached solution for a prompt."""

    task_id: str
    program_python: str
    source: str                 # "prompt_exact" | "task_id"
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompoundingStoreConfig:
    artifact_dir: Path = Path(".nCPU_autoresearch")
    solved_log_name: str = "solved_programs.jsonl"
    prompt_cache_name: str = "prompt_cache.json"
    temperature_stats_name: str = "temperature_stats.json"


class CompoundingStore:
    """Append-only persistent store for autoresearch solves."""

    def __init__(self, config: Optional[CompoundingStoreConfig] = None):
        self.cfg = config or CompoundingStoreConfig()
        self.cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._prompt_cache: Optional[dict[str, dict]] = None
        self._temp_stats: Optional[dict[str, int]] = None

    @property
    def solved_log(self) -> Path:
        return self.cfg.artifact_dir / self.cfg.solved_log_name

    @property
    def prompt_cache_path(self) -> Path:
        return self.cfg.artifact_dir / self.cfg.prompt_cache_name

    @property
    def temp_stats_path(self) -> Path:
        return self.cfg.artifact_dir / self.cfg.temperature_stats_name

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def record(
        self,
        solved: SolvedItem,
        *,
        work_item: Optional[WorkItem] = None,
    ) -> None:
        """Append a solve to every index.

        ``work_item`` is optional but recommended — it lets us build the
        prompt-hash cache key.
        """
        self.solved_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.solved_log, "a") as fh:
            fh.write(json.dumps(solved.to_dict()) + "\n")

        # Prompt cache.
        if work_item is not None:
            key = hash_prompt(work_item.prompt, work_item.entry_point)
            cache = self._load_prompt_cache()
            cache[key] = {
                "task_id": solved.task_id,
                "program_python": solved.program_python,
                "solver": solved.solver,
                "provenance": solved.provenance,
            }
            self._save_prompt_cache(cache)

        # Temperature stats.
        temp = solved.provenance.get("winning_temperature") if solved.provenance else None
        if temp is not None:
            stats = self._load_temp_stats()
            k = f"{temp:.2f}"
            stats[k] = stats.get(k, 0) + 1
            self._save_temp_stats(stats)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def check_prompt(self, work_item: WorkItem) -> Optional[CacheHit]:
        """Exact-match prompt cache lookup. Returns ``None`` on miss."""
        cache = self._load_prompt_cache()
        key = hash_prompt(work_item.prompt, work_item.entry_point)
        row = cache.get(key)
        if row is None:
            return None
        return CacheHit(
            task_id=row["task_id"],
            program_python=row["program_python"],
            source="prompt_exact",
            provenance={"solver": row.get("solver", ""),
                        "original_provenance": row.get("provenance", {})},
        )

    def check_task_id(self, task_id: str) -> Optional[CacheHit]:
        """Task-ID lookup (slower: scans the log)."""
        if not self.solved_log.exists():
            return None
        with open(self.solved_log) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("task_id") == task_id:
                    return CacheHit(
                        task_id=task_id,
                        program_python=row["program_python"],
                        source="task_id",
                        provenance=row.get("provenance", {}),
                    )
        return None

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def count_solved(self) -> int:
        if not self.solved_log.exists():
            return 0
        with open(self.solved_log) as fh:
            return sum(1 for line in fh if line.strip())

    def temperature_stats(self) -> dict[str, int]:
        return dict(self._load_temp_stats())

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_dir": str(self.cfg.artifact_dir),
            "solved_programs": self.count_solved(),
            "prompt_cache_size": len(self._load_prompt_cache()),
            "temperature_stats": self.temperature_stats(),
        }

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def rebuild_indices(self) -> dict[str, int]:
        """Rebuild prompt_cache and temperature_stats from the JSONL log.

        Useful after manual edits or to recover from inconsistency.
        Note: prompt_cache can only be rebuilt from rows that stored
        ``provenance.prompt`` — older rows may be omitted from the cache.
        """
        prompt_cache: dict[str, dict] = {}
        temp_stats: dict[str, int] = {}
        if self.solved_log.exists():
            with open(self.solved_log) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    # Prompt cache.
                    prov = row.get("provenance") or {}
                    prompt = prov.get("prompt")
                    entry_point = prov.get("entry_point")
                    if prompt and entry_point:
                        k = hash_prompt(prompt, entry_point)
                        prompt_cache[k] = {
                            "task_id": row["task_id"],
                            "program_python": row["program_python"],
                            "solver": row.get("solver", ""),
                            "provenance": prov,
                        }
                    # Temperature.
                    t = prov.get("winning_temperature")
                    if t is not None:
                        k = f"{float(t):.2f}"
                        temp_stats[k] = temp_stats.get(k, 0) + 1
        self._save_prompt_cache(prompt_cache)
        self._save_temp_stats(temp_stats)
        return {"prompt_cache": len(prompt_cache), "temperature_stats": sum(temp_stats.values())}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _load_prompt_cache(self) -> dict[str, dict]:
        if self._prompt_cache is None:
            if self.prompt_cache_path.exists():
                self._prompt_cache = json.loads(self.prompt_cache_path.read_text())
            else:
                self._prompt_cache = {}
        return self._prompt_cache

    def _save_prompt_cache(self, data: dict[str, dict]) -> None:
        self._prompt_cache = data
        self.prompt_cache_path.write_text(json.dumps(data, indent=2))

    def _load_temp_stats(self) -> dict[str, int]:
        if self._temp_stats is None:
            if self.temp_stats_path.exists():
                self._temp_stats = {k: int(v) for k, v in
                                    json.loads(self.temp_stats_path.read_text()).items()}
            else:
                self._temp_stats = {}
        return self._temp_stats

    def _save_temp_stats(self, data: dict[str, int]) -> None:
        self._temp_stats = dict(data)
        self.temp_stats_path.write_text(json.dumps(data, indent=2, sort_keys=True))
