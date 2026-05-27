"""Task-session persistence for `ArrayProgramLibrary` (N1).

When an LLM inference session finishes, the `HFTaskLocalFastWeightsProvider`
resets its low-rank adapters — that's the correct behavior for *weights*,
which are task-local and must not leak across tasks. But **programs** (the
M3 cached skills) are a different kind of memory: they are task-agnostic
computation recipes that should persist across sessions, the way
`~/.nsynth_learned_biases.jsonl` persists across Rust-synthesis runs.

`ProgramLibrarySession` provides that persistence:

* `begin_task(name)` — attach/reuse the library, load from disk if present.
* `library` — the `ArrayProgramLibrary` the provider should consult.
* `apply_converged_program(...)` — forward through the thought head's
  `consult_library`, auto-caching any converged programs.
* `end_task()` — persist the library back to disk.

The session is deliberately *not* coupled to `HFTaskLocalFastWeightsProvider`
so it can be:

* Unit-tested without an HF model download.
* Embedded inside any other provider (local, remote, synthetic) without
  pulling in the HF code path.
* Shared across providers: the same on-disk library file can be opened by a
  Qwen-based provider and a Llama-based provider as long as they agree on
  the hidden-state signature convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    ArrayThoughtLibraryResult,
)


DEFAULT_LIBRARY_PATH: Path = Path.home() / ".nCPU_program_library.json"


@dataclass
class ProgramLibrarySessionConfig:
    """Configuration for `ProgramLibrarySession`."""

    library_path: Optional[Path] = None
    create_if_missing: bool = True
    save_on_end_task: bool = True
    library_config: Optional[ArrayProgramLibraryConfig] = None
    convergence_gap_threshold: float = 0.15
    auto_cache: bool = True

    def resolved_path(self) -> Path:
        return Path(self.library_path or DEFAULT_LIBRARY_PATH).expanduser()


@dataclass
class ProgramLibraryTaskSummary:
    """Summary returned from `end_task`."""

    task_name: Optional[str]
    library_path: str
    entry_count: int
    total_hits: int
    newly_cached_count: int
    saved: bool
    diff: Optional[dict[str, Any]] = None


class ProgramLibrarySession:
    """Attach an `ArrayProgramLibrary` to a task lifecycle with disk persistence."""

    def __init__(self, config: Optional[ProgramLibrarySessionConfig] = None):
        self.config = config or ProgramLibrarySessionConfig()
        self._library: Optional[ArrayProgramLibrary] = None
        self._active_task: Optional[str] = None
        self._task_newly_cached: int = 0
        self._begin_snapshot: Optional[list[dict[str, Any]]] = None

    @property
    def library(self) -> ArrayProgramLibrary:
        if self._library is None:
            raise RuntimeError(
                "begin_task() must be called before accessing the library"
            )
        return self._library

    def begin_task(
        self,
        task_name: str,
        *,
        force_reload: bool = False,
    ) -> dict[str, Any]:
        """Load the library from disk (if present) and mark the task active."""
        path = self.config.resolved_path()
        reused_from_disk = False
        if force_reload or self._library is None:
            if path.exists():
                self._library = ArrayProgramLibrary.load(path)
                reused_from_disk = True
            elif self.config.create_if_missing:
                self._library = ArrayProgramLibrary(
                    self.config.library_config
                    or ArrayProgramLibraryConfig()
                )
            else:
                raise FileNotFoundError(
                    f"library file {path} not found and create_if_missing=False"
                )
        self._active_task = task_name
        self._task_newly_cached = 0
        # Snapshot at task start so end_task() can compute a before/after
        # diff. Empty list is fine for a fresh library.
        self._begin_snapshot = self._library.snapshot()
        return {
            "task_name": task_name,
            "library_path": str(path),
            "entry_count": len(self._library),
            "reused_from_disk": reused_from_disk,
        }

    def apply_converged_program(
        self,
        thought_head: Any,
        hidden_state: torch.Tensor,
        array_inputs: torch.Tensor,
        *,
        lengths: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> ArrayThoughtLibraryResult:
        """Forward through `thought_head.consult_library` for the active task."""
        if self._library is None or self._active_task is None:
            raise RuntimeError("begin_task() must be called first")
        result = thought_head.consult_library(
            hidden_state,
            array_inputs,
            self._library,
            lengths=lengths,
            temperature=temperature,
            auto_cache=self.config.auto_cache,
            convergence_gap_threshold=self.config.convergence_gap_threshold,
            task_name=self._active_task,
        )
        self._task_newly_cached += sum(
            1 for flag in result.newly_cached if flag
        )
        return result

    def end_task(
        self,
        *,
        save: Optional[bool] = None,
        include_diff: bool = True,
    ) -> ProgramLibraryTaskSummary:
        """Persist the library to disk and return a summary with diff."""
        if self._library is None:
            raise RuntimeError("begin_task() must be called first")
        resolved_save = self.config.save_on_end_task if save is None else save
        path = self.config.resolved_path()
        total_hits = sum(entry.hit_count for entry in self._library.entries)
        diff: Optional[dict[str, Any]] = None
        if include_diff and self._begin_snapshot is not None:
            diff = self._library.diff_against(self._begin_snapshot)
        if resolved_save:
            self._library.save(path)
        summary = ProgramLibraryTaskSummary(
            task_name=self._active_task,
            library_path=str(path),
            entry_count=len(self._library),
            total_hits=int(total_hits),
            newly_cached_count=int(self._task_newly_cached),
            saved=resolved_save,
            diff=diff,
        )
        self._active_task = None
        self._task_newly_cached = 0
        self._begin_snapshot = None
        return summary


def attach_session_to_provider(
    provider: Any,
    config: Optional[ProgramLibrarySessionConfig] = None,
) -> ProgramLibrarySession:
    """Attach a `ProgramLibrarySession` to any provider with begin/end hooks.

    The provider must expose `begin_task`, `end_task`, and a
    `program_library_session` attribute slot. This helper installs the
    session, wraps the provider's task-lifecycle methods so the library is
    loaded/saved automatically, and returns the session for direct
    interaction by the caller.
    """
    session = ProgramLibrarySession(config)
    if not hasattr(provider, "begin_task") or not hasattr(provider, "end_task"):
        raise TypeError(
            "provider must expose begin_task and end_task for session attachment"
        )
    original_begin = provider.begin_task
    original_end = provider.end_task

    def wrapped_begin(task_name: str, *args: Any, **kwargs: Any) -> Any:
        session.begin_task(task_name)
        return original_begin(task_name, *args, **kwargs)

    def wrapped_end(*args: Any, **kwargs: Any) -> Any:
        summary = session.end_task()
        base = original_end(*args, **kwargs)
        if isinstance(base, dict):
            base["program_library"] = {
                "entries": summary.entry_count,
                "total_hits": summary.total_hits,
                "newly_cached": summary.newly_cached_count,
                "library_path": summary.library_path,
                "saved": summary.saved,
            }
        return base

    provider.program_library_session = session
    provider.begin_task = wrapped_begin
    provider.end_task = wrapped_end
    return session


__all__ = [
    "DEFAULT_LIBRARY_PATH",
    "ProgramLibrarySession",
    "ProgramLibrarySessionConfig",
    "ProgramLibraryTaskSummary",
    "attach_session_to_provider",
]
