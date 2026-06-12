"""Task-local fast-weight program library for NPCoT milestone M3.

The M2 `ArrayExecutableThoughtHead` emits a soft (differentiable) reduction
program whose every choice is controlled by the hidden state. Once a hidden
state has driven such a program to convergence — i.e. the argmax (discrete)
program closely matches the soft program's output — the program becomes a
*reusable skill*: a closed-form procedure that the head does not need to
re-solve by gradient descent on the next visit.

This module captures that insight:

* `DiscreteArrayProgram` is the discretized argmax of a soft program, with its
  own pure-tensor execution path (no autograd, no softmax, no temperature).
* `ArrayProgramLibrary` is a cosine-similarity keyed cache of those programs
  indexed by hidden-state signature. It exposes `lookup`, `record`, JSON
  persistence, and capacity-bounded LRU-style eviction.

The library is consumed by `ArrayExecutableThoughtHead.consult_library`: on a
hit, the head short-circuits to the discrete program; on a miss, it runs the
soft forward and — if soft/discrete agreement is tight — caches the freshly
synthesized program for next time. This is the "skill accumulation" claim of
the NPCoT loop: reasoning converges into a vocabulary of provably reusable
programs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

import torch

from ncpu.self_optimizing.array_executable_thought_head import (
    _ELEM_TRANSFORMS,
    _INIT_CHOICES,
    _LOG_EPS,
    _NEG_LARGE,
    _POS_LARGE,
    _POST_SCALES,
    _REDUCE_OPS,
)


# Mirrors `_INIT_CHOICES` index-for-index: "0", "1", "-large", "+large".
# `+large` is the positive-infinity proxy that lets a `min` reduce start above
# every realistic element. Append-only — indices 0-2 are frozen so existing
# library JSONs keep their meaning.
_INIT_VALUES: tuple[float, ...] = (0.0, 1.0, _NEG_LARGE, _POS_LARGE)


def _try_load_native_backend() -> Optional[Any]:
    """Locate the optional Rust+Metal npcot_exec backend.

    Loads `kernels/rust_metal/ncpu_metal.abi3.so` directly (without polluting
    sys.path or importing the venv-installed torch) and returns the module,
    or None if the extension cannot be loaded (e.g. non-macOS, no compiled
    `.so`, missing Metal device).
    """
    import importlib.util
    import sys

    so_path = (
        Path(__file__).resolve().parents[2]
        / "kernels"
        / "rust_metal"
        / "ncpu_metal.abi3.so"
    )
    if not so_path.exists():
        return None
    try:
        # The PyO3 abi3 extension's init function is named
        # `PyInit_ncpu_metal` (matching the pymodule declaration), so we
        # must load it under that exact module name.
        previous = sys.modules.get("ncpu_metal")
        spec = importlib.util.spec_from_file_location("ncpu_metal", so_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules["ncpu_metal"] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            # Restore previous registration so we don't break other consumers.
            if previous is not None:
                sys.modules["ncpu_metal"] = previous
            else:
                sys.modules.pop("ncpu_metal", None)
            return None
        if not hasattr(module, "npcot_execute_cpu"):
            return None
        return module
    except Exception:
        return None


_NATIVE_BACKEND_CACHE: dict[str, Any] = {}


def get_native_backend() -> Optional[Any]:
    """Lazy-load the native backend (cached)."""
    if "module" not in _NATIVE_BACKEND_CACHE:
        _NATIVE_BACKEND_CACHE["module"] = _try_load_native_backend()
    return _NATIVE_BACKEND_CACHE["module"]


def reset_native_backend_cache() -> None:
    """Clear the native backend cache — useful for tests."""
    _NATIVE_BACKEND_CACHE.clear()


def _apply_transform(x: torch.Tensor, idx: int) -> torch.Tensor:
    if idx == 0:
        return x
    if idx == 1:
        return x * x
    if idx == 2:
        return x.abs()
    if idx == 3:
        return torch.ones_like(x)
    if idx == 4:
        return (x > 0).to(x.dtype)
    if idx == 5:
        # log(|x| + eps) — numerically stable operand for log-domain product.
        return torch.log(x.abs() + _LOG_EPS)
    raise ValueError(f"unknown transform index {idx}")


def _apply_reduce(acc: torch.Tensor, f: torch.Tensor, idx: int) -> torch.Tensor:
    if idx == 0:
        return acc + f
    if idx == 1:
        return acc * f
    if idx == 2:
        return torch.maximum(acc, f)
    if idx == 3:
        return torch.minimum(acc, f)
    raise ValueError(f"unknown reduce index {idx}")


@dataclass
class DiscreteArrayProgram:
    """Argmax of a soft array-thought program: a pure, reusable skill."""

    init_idx: int
    transform_idx: int
    reduce_idx: int
    post_scale_idx: int
    offset: float

    def __post_init__(self) -> None:
        if not 0 <= self.init_idx < len(_INIT_CHOICES):
            raise ValueError(f"init_idx out of range: {self.init_idx}")
        if not 0 <= self.transform_idx < len(_ELEM_TRANSFORMS):
            raise ValueError(f"transform_idx out of range: {self.transform_idx}")
        if not 0 <= self.reduce_idx < len(_REDUCE_OPS):
            raise ValueError(f"reduce_idx out of range: {self.reduce_idx}")
        if not 0 <= self.post_scale_idx < len(_POST_SCALES):
            raise ValueError(f"post_scale_idx out of range: {self.post_scale_idx}")
        self.offset = float(self.offset)

    @property
    def init_value(self) -> float:
        return _INIT_VALUES[self.init_idx]

    @property
    def init_label(self) -> str:
        return _INIT_CHOICES[self.init_idx]

    @property
    def transform_label(self) -> str:
        return _ELEM_TRANSFORMS[self.transform_idx]

    @property
    def reduce_label(self) -> str:
        return _REDUCE_OPS[self.reduce_idx]

    @property
    def post_scale_label(self) -> str:
        return _POST_SCALES[self.post_scale_idx]

    def execute(self, arrays: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Run the discrete program over a batch of arrays. No gradients needed."""
        if arrays.ndim != 2:
            raise ValueError(f"arrays must be rank-2, got shape {tuple(arrays.shape)}")
        batch_size, max_len = arrays.shape
        if lengths.shape != (batch_size,):
            raise ValueError(
                f"lengths must be shape ({batch_size},), got {tuple(lengths.shape)}"
            )
        device = arrays.device
        dtype = arrays.dtype if arrays.is_floating_point() else torch.float32
        arrays_f = arrays.to(dtype=dtype)
        lengths_f = lengths.to(device=device, dtype=dtype)

        acc = torch.full(
            (batch_size,),
            float(self.init_value),
            device=device,
            dtype=dtype,
        )
        positions = torch.arange(max_len, device=device).unsqueeze(0)
        active_mask = positions < lengths_f.unsqueeze(1)

        for i in range(max_len):
            x_i = arrays_f[:, i]
            active = active_mask[:, i]
            f_i = _apply_transform(x_i, self.transform_idx)
            new_acc = _apply_reduce(acc, f_i, self.reduce_idx)
            acc = torch.where(active, new_acc, acc)

        if self.post_scale_idx == 0:
            post_value = acc
        elif self.post_scale_idx == 1:
            denom = torch.clamp(lengths_f, min=1.0)
            post_value = acc / denom
        else:
            # idx == 2: exp(acc) — stable product-recovery. Clamp input so
            # float32 doesn't overflow on spurious large accumulators.
            post_value = torch.exp(torch.clamp(acc, min=-30.0, max=30.0))
        return post_value + float(self.offset)

    def execute_native(
        self,
        arrays: torch.Tensor,
        lengths: torch.Tensor,
        *,
        backend: str = "auto",
    ) -> torch.Tensor:
        """Run the discrete program via the native Rust/Metal backend.

        `backend`:
        * `"auto"` (default) — prefer Metal if a GPU executor is available,
          fall back to native Rust CPU, then to the pure-Python reference.
        * `"rust_cpu"` — force the pure-Rust CPU path.
        * `"metal"` — force the Metal GPU path (raises if unavailable).
        * `"python"` — bypass native backends entirely (same as `execute`).

        Returns a CPU tensor of shape `(batch,)` with the discrete results.
        Use this when you want to skip PyTorch overhead on a hot library-hit
        path — typically 3-10x faster than `execute()` on modest batches.
        """
        if arrays.ndim != 2:
            raise ValueError(f"arrays must be rank-2, got {tuple(arrays.shape)}")
        batch, max_len = arrays.shape
        if lengths.shape != (batch,):
            raise ValueError(
                f"lengths must be shape ({batch},), got {tuple(lengths.shape)}"
            )

        if backend == "python":
            return self.execute(arrays, lengths)

        native = get_native_backend()
        if native is None:
            if backend != "auto":
                raise RuntimeError(
                    f"native backend unavailable; cannot satisfy backend={backend!r}"
                )
            return self.execute(arrays, lengths)

        programs_flat = [
            float(self.init_idx),
            float(self.transform_idx),
            float(self.reduce_idx),
            float(self.post_scale_idx),
            float(self.offset),
        ]
        arrays_flat = arrays.detach().to(torch.float32).cpu().flatten().tolist()
        lengths_list = [int(v) for v in lengths.detach().cpu().tolist()]

        if backend in ("auto", "metal"):
            if hasattr(native, "NpcotGpuExecutor"):
                try:
                    executor = native.NpcotGpuExecutor()
                    result = executor.execute(
                        programs_flat, arrays_flat, lengths_list, int(max_len)
                    )
                    return torch.tensor(result, dtype=torch.float32)
                except Exception:
                    if backend == "metal":
                        raise
        if backend in ("auto", "rust_cpu"):
            result = native.npcot_execute_cpu(
                programs_flat, arrays_flat, lengths_list, int(max_len)
            )
            return torch.tensor(result, dtype=torch.float32)

        raise ValueError(f"unknown backend: {backend}")

    def render(self) -> str:
        # Transform label substitution — `x` → `arr[i]` for the identity,
        # `x*x` → `arr[i]*arr[i]`, `|x|` → `|arr[i]|`, etc. The `log|x|`
        # transform renders as `ln(|arr[i]| + eps)`.
        if self.transform_label == "log|x|":
            trans_label = "ln(|arr[i]| + eps)"
        else:
            trans_label = self.transform_label.replace("x", "arr[i]")
        reduce_label = self.reduce_label
        body_line = {
            "+": f"acc += {trans_label}",
            "*": f"acc *= {trans_label}",
            "max": f"acc = max(acc, {trans_label})",
            "min": f"acc = min(acc, {trans_label})",
        }[reduce_label]
        offset_str = f" + {self.offset:.3f}" if abs(self.offset) > 1e-3 else ""
        if self.post_scale_label == "acc":
            return_expr = "acc"
        elif self.post_scale_label == "acc/len":
            return_expr = "acc / max(len(arr), 1)"
        else:
            return_expr = "exp(clamp(acc, -30, 30))"
        return (
            "fn array_thought(arr: &[i64]) -> f64 {\n"
            f"    let mut acc: f64 = {self.init_label};\n"
            "    for i in 0..arr.len() {\n"
            f"        {body_line};\n"
            "    }\n"
            f"    return {return_expr}{offset_str};\n"
            "}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "init_idx": int(self.init_idx),
            "transform_idx": int(self.transform_idx),
            "reduce_idx": int(self.reduce_idx),
            "post_scale_idx": int(self.post_scale_idx),
            "offset": float(self.offset),
            "program_text": self.render(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiscreteArrayProgram":
        return cls(
            init_idx=int(payload["init_idx"]),
            transform_idx=int(payload["transform_idx"]),
            reduce_idx=int(payload["reduce_idx"]),
            post_scale_idx=int(payload["post_scale_idx"]),
            offset=float(payload["offset"]),
        )

    @classmethod
    def from_soft_distributions(
        cls,
        distributions: dict[str, torch.Tensor],
        batch_index: int,
    ) -> "DiscreteArrayProgram":
        """Take argmax of each parameter distribution to produce a hard program."""
        return cls(
            init_idx=int(torch.argmax(distributions["init"][batch_index]).item()),
            transform_idx=int(torch.argmax(distributions["transform"][batch_index]).item()),
            reduce_idx=int(torch.argmax(distributions["reduce"][batch_index]).item()),
            post_scale_idx=int(torch.argmax(distributions["post_scale"][batch_index]).item()),
            offset=float(distributions["post_offset"][batch_index].item()),
        )

    def key(self) -> tuple[int, int, int, int]:
        """Structural key (ignores offset) for quick dedup."""
        return (
            self.init_idx,
            self.transform_idx,
            self.reduce_idx,
            self.post_scale_idx,
        )


@dataclass
class ArrayProgramLibraryConfig:
    """Configuration for the cosine-similarity keyed program cache."""

    similarity_threshold: float = 0.92
    max_entries: int = 128
    normalize_epsilon: float = 1e-8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LibraryEntry:
    """One cached program, indexed by a unit-norm hidden-state signature."""

    signature: list[float]
    program: DiscreteArrayProgram
    hit_count: int = 0
    task_name: Optional[str] = None
    cached_at_step: Optional[int] = None
    convergence_gap: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": [float(v) for v in self.signature],
            "program": self.program.to_dict(),
            "hit_count": int(self.hit_count),
            "task_name": self.task_name,
            "cached_at_step": self.cached_at_step,
            "convergence_gap": (
                float(self.convergence_gap) if self.convergence_gap is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LibraryEntry":
        return cls(
            signature=[float(v) for v in payload["signature"]],
            program=DiscreteArrayProgram.from_dict(payload["program"]),
            hit_count=int(payload.get("hit_count", 0)),
            task_name=payload.get("task_name"),
            cached_at_step=payload.get("cached_at_step"),
            convergence_gap=payload.get("convergence_gap"),
        )


class ArrayProgramLibrary:
    """Cosine-similarity keyed library of discrete array-reduction programs."""

    def __init__(self, config: Optional[ArrayProgramLibraryConfig] = None):
        self.config = config or ArrayProgramLibraryConfig()
        self._entries: list[LibraryEntry] = []
        self._native_index: Optional[Any] = None

    def build_native_index(self) -> bool:
        """Build an optional Rust-backed sharded lookup index.

        Returns True if the native index was built successfully, False if
        the native backend is unavailable. Call after `record()`-ing a
        sizable number of entries — for libraries with <50 entries the
        Python scan is already fast enough.
        """
        native = get_native_backend()
        if native is None or not hasattr(native, "NpcotLibraryIndex"):
            self._native_index = None
            return False
        index = native.NpcotLibraryIndex(
            float(self.config.similarity_threshold)
        )
        for entry in self._entries:
            program = entry.program
            params = [
                float(program.init_idx),
                float(program.transform_idx),
                float(program.reduce_idx),
                float(program.post_scale_idx),
                float(program.offset),
            ]
            index.insert(list(entry.signature), params)
        self._native_index = index
        return True

    def drop_native_index(self) -> None:
        self._native_index = None

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[LibraryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    @staticmethod
    def _normalize_signature(hidden: torch.Tensor, eps: float) -> list[float]:
        if hidden.ndim != 1:
            raise ValueError(
                f"hidden signature must be rank-1, got shape {tuple(hidden.shape)}"
            )
        flat = hidden.detach().to(torch.float32).cpu()
        norm = float(torch.linalg.norm(flat).item())
        if norm < eps:
            return [0.0] * int(flat.numel())
        return (flat / norm).tolist()

    def _score(
        self,
        query_signature: list[float],
        entry_signature: list[float],
    ) -> float:
        if len(query_signature) != len(entry_signature):
            return -1.0
        total = 0.0
        for a, b in zip(query_signature, entry_signature):
            total += a * b
        return total

    def lookup(self, hidden_state: torch.Tensor) -> Optional[LibraryEntry]:
        """Return the best matching entry above `similarity_threshold`, or None."""
        if not self._entries:
            return None
        query = self._normalize_signature(hidden_state, self.config.normalize_epsilon)
        if all(value == 0.0 for value in query):
            return None
        # Fast path: if a native sharded index has been built, consult it
        # first. It returns program params; we still return the original
        # LibraryEntry so callers see hit_count bookkeeping.
        if self._native_index is not None:
            result = self._native_index.lookup(list(query))
            if result is not None:
                params, _similarity = result
                init_idx, transform_idx, reduce_idx, post_scale_idx, offset = params
                key = (
                    int(init_idx),
                    int(transform_idx),
                    int(reduce_idx),
                    int(post_scale_idx),
                )
                for entry in self._entries:
                    if entry.program.key() == key and abs(
                        entry.program.offset - float(offset)
                    ) < 1e-6:
                        entry.hit_count += 1
                        return entry
        best_entry: Optional[LibraryEntry] = None
        best_score: float = -1.0
        for entry in self._entries:
            score = self._score(query, entry.signature)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry is not None and best_score >= self.config.similarity_threshold:
            best_entry.hit_count += 1
            return best_entry
        return None

    def record(
        self,
        hidden_state: torch.Tensor,
        program: DiscreteArrayProgram,
        *,
        task_name: Optional[str] = None,
        cached_at_step: Optional[int] = None,
        convergence_gap: Optional[float] = None,
    ) -> LibraryEntry:
        """Insert or refresh an entry. Near-duplicates overwrite their program."""
        signature = self._normalize_signature(
            hidden_state, self.config.normalize_epsilon
        )
        if all(value == 0.0 for value in signature):
            raise ValueError("cannot record zero-norm hidden-state signature")

        best_entry: Optional[LibraryEntry] = None
        best_score: float = -1.0
        for entry in self._entries:
            score = self._score(signature, entry.signature)
            if score > best_score:
                best_score = score
                best_entry = entry

        if (
            best_entry is not None
            and best_score >= self.config.similarity_threshold
        ):
            best_entry.program = program
            best_entry.signature = signature
            if task_name is not None:
                best_entry.task_name = task_name
            if cached_at_step is not None:
                best_entry.cached_at_step = cached_at_step
            if convergence_gap is not None:
                best_entry.convergence_gap = float(convergence_gap)
            return best_entry

        entry = LibraryEntry(
            signature=signature,
            program=program,
            hit_count=0,
            task_name=task_name,
            cached_at_step=cached_at_step,
            convergence_gap=(
                float(convergence_gap) if convergence_gap is not None else None
            ),
        )
        self._entries.append(entry)
        if len(self._entries) > self.config.max_entries:
            self._entries.sort(key=lambda e: e.hit_count, reverse=True)
            del self._entries[self.config.max_entries :]
        return entry

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "entries": [entry.to_dict() for entry in self._entries],
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ArrayProgramLibrary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config_dict = payload.get("config") or {}
        config = ArrayProgramLibraryConfig(**config_dict)
        library = cls(config)
        for entry_dict in payload.get("entries", []):
            library._entries.append(LibraryEntry.from_dict(entry_dict))
        return library

    def fingerprint(self) -> str:
        """Content-addressable hash of the library's stable state.

        Two libraries fingerprint identically iff they hold the same set of
        (signature, program) pairs. Ordering, hit counts, cached-at-step
        and convergence-gap metadata are all excluded — those are
        deployment-time mutable. Useful for:

        * Recruiting: a firm can share a library fingerprint publicly to
          prove a candidate is working against the same reasoning skills
          without exposing the library itself.
        * Reproducibility: two training runs that produce the same skill
          set hash identically, even if their entries are in different
          insertion order.
        * Library indexing: a registry can use fingerprints as canonical
          IDs.
        """
        import hashlib

        canonical_entries = []
        for entry in self._entries:
            canonical_entries.append(
                {
                    "signature": [float(v) for v in entry.signature],
                    "program": entry.program.to_dict(),
                }
            )
        canonical_entries.sort(
            key=lambda e: json.dumps(e["signature"], sort_keys=True)
            + json.dumps(e["program"], sort_keys=True)
        )
        payload = json.dumps(
            {
                "schema_version": "npcot/1.0",
                "similarity_threshold": self.config.similarity_threshold,
                "entries": canonical_entries,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"npcot1:{digest[:32]}"

    def audit_report(self) -> dict[str, Any]:
        """Return a JSON-serializable compliance view of the library.

        The report is the minimum a regulated-industry auditor needs: every
        skill the model has accumulated, how many times it's been used, how
        close the discrete program was to its soft parent when cached, and
        the exact Rust-like pseudocode that will run on each library hit.
        """
        total_hits = sum(entry.hit_count for entry in self._entries)
        gaps = [
            entry.convergence_gap
            for entry in self._entries
            if entry.convergence_gap is not None
        ]
        program_keys = {entry.program.key() for entry in self._entries}

        return {
            "summary": {
                "entry_count": len(self._entries),
                "unique_program_shapes": len(program_keys),
                "total_hits": int(total_hits),
                "avg_convergence_gap": (
                    float(sum(gaps) / len(gaps)) if gaps else None
                ),
                "max_convergence_gap": float(max(gaps)) if gaps else None,
                "config": self.config.to_dict(),
            },
            "entries": [
                {
                    "task_name": entry.task_name,
                    "program": entry.program.to_dict(),
                    "hit_count": int(entry.hit_count),
                    "convergence_gap": (
                        float(entry.convergence_gap)
                        if entry.convergence_gap is not None
                        else None
                    ),
                    "cached_at_step": entry.cached_at_step,
                    "signature_dim": len(entry.signature),
                }
                for entry in self._entries
            ],
        }

    def snapshot(self) -> list[dict[str, Any]]:
        """Capture a lightweight, JSON-serializable state snapshot.

        The snapshot is the ordered list of entry dicts — same shape as what
        `to_dict` would produce — intended to be handed to `diff_against`
        later to compute added/removed/changed skills.
        """
        return [entry.to_dict() for entry in self._entries]

    def diff_against(
        self, snapshot: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compare the current library state to a prior snapshot.

        Returns a report with three buckets — `added` (new entries since
        snapshot), `removed` (entries present in snapshot but no longer
        here), and `changed` (program or task_name modified for the same
        signature). Structurally-identical entries whose only change is
        `hit_count` count as `unchanged` — they've been *used*, not
        *modified*, which is the distinction auditors care about.
        """
        current = {
            self._entry_fingerprint(entry): entry for entry in self._entries
        }
        previous: dict[str, dict[str, Any]] = {}
        for item in snapshot:
            fingerprint = self._snapshot_fingerprint(item)
            previous[fingerprint] = item

        added = [
            current[fp].to_dict()
            for fp in current.keys() - previous.keys()
        ]
        removed = [
            previous[fp] for fp in previous.keys() - current.keys()
        ]
        changed: list[dict[str, Any]] = []
        unchanged = 0
        hits_since_snapshot = 0
        for fp in current.keys() & previous.keys():
            curr = current[fp]
            prev = previous[fp]
            prev_hits = int(prev.get("hit_count", 0))
            hits_since_snapshot += max(curr.hit_count - prev_hits, 0)
            program_changed = (
                curr.program.to_dict() != prev.get("program")
            )
            task_changed = curr.task_name != prev.get("task_name")
            if program_changed or task_changed:
                changed.append(
                    {
                        "signature": list(curr.signature),
                        "before": prev,
                        "after": curr.to_dict(),
                    }
                )
            else:
                unchanged += 1
        return {
            "added": added,
            "removed": removed,
            "changed": changed,
            "unchanged": unchanged,
            "hits_since_snapshot": int(hits_since_snapshot),
        }

    @staticmethod
    def _entry_fingerprint(entry: "LibraryEntry") -> str:
        sig_str = ",".join(f"{v:.6f}" for v in entry.signature)
        return sig_str

    @staticmethod
    def _snapshot_fingerprint(snapshot_entry: dict[str, Any]) -> str:
        sig = snapshot_entry.get("signature", [])
        return ",".join(f"{float(v):.6f}" for v in sig)

    def audit_markdown(self) -> str:
        """Return a human-readable markdown audit of every cached skill."""
        report = self.audit_report()
        summary = report["summary"]
        lines: list[str] = [
            "# Array Program Library — Audit Report",
            "",
            "## Summary",
            "",
            f"- **Entries**: {summary['entry_count']}",
            f"- **Unique program shapes**: {summary['unique_program_shapes']}",
            f"- **Total library hits**: {summary['total_hits']}",
        ]
        avg_gap = summary["avg_convergence_gap"]
        max_gap = summary["max_convergence_gap"]
        if avg_gap is not None:
            lines.append(f"- **Avg convergence gap**: {avg_gap:.4f}")
        if max_gap is not None:
            lines.append(f"- **Max convergence gap**: {max_gap:.4f}")
        lines.append(
            f"- **Similarity threshold**: "
            f"{summary['config']['similarity_threshold']:.3f}"
        )
        lines.append(f"- **Capacity**: {summary['config']['max_entries']} entries")

        lines.extend(["", "## Cached Skills", ""])
        if not report["entries"]:
            lines.append("_(library is empty)_")
            return "\n".join(lines)

        for index, entry_report in enumerate(report["entries"], start=1):
            task_name = entry_report["task_name"] or "(unnamed)"
            lines.append(f"### Skill {index}: `{task_name}`")
            lines.append("")
            lines.append(f"- Hits: **{entry_report['hit_count']}**")
            gap = entry_report["convergence_gap"]
            if gap is not None:
                lines.append(f"- Convergence gap when cached: {gap:.4f}")
            cached_at = entry_report["cached_at_step"]
            if cached_at is not None:
                lines.append(f"- Cached at step: {cached_at}")
            lines.append(f"- Signature dim: {entry_report['signature_dim']}")
            lines.append("")
            lines.append("```rust")
            lines.append(entry_report["program"]["program_text"])
            lines.append("```")
            lines.append("")
        return "\n".join(lines)


def merge_libraries(
    libraries: list[ArrayProgramLibrary],
    *,
    conflict_resolution: str = "keep_more_hits",
    target_config: Optional[ArrayProgramLibraryConfig] = None,
) -> ArrayProgramLibrary:
    """Merge a list of `ArrayProgramLibrary` into one.

    `conflict_resolution`:
    * `"keep_more_hits"` — if two entries have similarity ≥ threshold,
      keep the one with more hits (the one we have more evidence is used).
    * `"keep_newer"` — keep the entry from the later-in-list library.
    * `"keep_both"` — keep both, ignoring the similarity dedup.

    Useful for federated-reasoning scenarios: two organizations training
    NPCoT in the same domain can merge their libraries via a trusted
    broker, and the broker can distribute the union back to both parties
    without either needing access to the other's training data.

    The resulting library is assumed to share signature dim; mixed-dim
    entries are rejected with a ValueError.
    """
    if not libraries:
        return ArrayProgramLibrary(target_config or ArrayProgramLibraryConfig())
    if conflict_resolution not in ("keep_more_hits", "keep_newer", "keep_both"):
        raise ValueError(
            f"unknown conflict_resolution: {conflict_resolution!r}"
        )

    resolved_config = (
        target_config
        or ArrayProgramLibraryConfig(
            similarity_threshold=min(
                lib.config.similarity_threshold for lib in libraries
            ),
            max_entries=max(
                lib.config.max_entries for lib in libraries
            ),
        )
    )
    merged = ArrayProgramLibrary(resolved_config)

    signature_dim: Optional[int] = None
    for library in libraries:
        for entry in library.entries:
            if signature_dim is None:
                signature_dim = len(entry.signature)
            elif signature_dim != len(entry.signature):
                raise ValueError(
                    f"signature dim mismatch in merge: "
                    f"{signature_dim} vs {len(entry.signature)}"
                )
            # Scan existing merged entries for a near-match.
            best: Optional[LibraryEntry] = None
            best_score = -1.0
            for existing in merged._entries:
                score = _cosine(entry.signature, existing.signature)
                if score > best_score:
                    best_score = score
                    best = existing

            if (
                conflict_resolution != "keep_both"
                and best is not None
                and best_score >= resolved_config.similarity_threshold
            ):
                # Dedup: decide which of (existing, incoming) survives.
                if conflict_resolution == "keep_newer":
                    best.program = entry.program
                    best.task_name = entry.task_name or best.task_name
                    best.hit_count = max(best.hit_count, entry.hit_count)
                    best.convergence_gap = (
                        entry.convergence_gap
                        if entry.convergence_gap is not None
                        else best.convergence_gap
                    )
                elif conflict_resolution == "keep_more_hits":
                    if entry.hit_count > best.hit_count:
                        best.program = entry.program
                        best.task_name = entry.task_name or best.task_name
                        best.hit_count = entry.hit_count
                        best.convergence_gap = (
                            entry.convergence_gap
                            if entry.convergence_gap is not None
                            else best.convergence_gap
                        )
                continue

            merged._entries.append(
                LibraryEntry(
                    signature=[float(v) for v in entry.signature],
                    program=entry.program,
                    hit_count=int(entry.hit_count),
                    task_name=entry.task_name,
                    cached_at_step=entry.cached_at_step,
                    convergence_gap=entry.convergence_gap,
                )
            )

    if len(merged._entries) > resolved_config.max_entries:
        merged._entries.sort(key=lambda e: e.hit_count, reverse=True)
        del merged._entries[resolved_config.max_entries :]
    return merged


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    a_norm = sum(x * x for x in a) ** 0.5
    b_norm = sum(x * x for x in b) ** 0.5
    denom = a_norm * b_norm
    if denom < 1e-9:
        return -1.0
    return dot / denom


def transfer_library(
    source: ArrayProgramLibrary,
    *,
    projection: torch.Tensor,
    target_config: Optional[ArrayProgramLibraryConfig] = None,
    task_name_suffix: str = "_xfer",
) -> ArrayProgramLibrary:
    """Reproject an `ArrayProgramLibrary` onto a new hidden-dim space.

    `projection` is a rank-2 tensor of shape `(target_dim, source_dim)`. Each
    entry's signature is matrix-multiplied by this projection and then
    re-normalized to unit L2 norm. Programs themselves (the reusable skills)
    carry over unchanged — they operate on array inputs, not on hidden states,
    so they remain valid in the target model's execution environment.

    Use cases:
    * Student distillation: carry a teacher's learned skill library into a
      smaller student that was trained on the same tasks.
    * Hidden-dim architecture change: move a library from a hidden_dim=512
      checkpoint into a hidden_dim=768 checkpoint via a learned map.
    * Vocabulary sharing across training runs: the program vocabulary is
      architecture-agnostic; the signature is the only model-specific part.
    """
    if projection.ndim != 2:
        raise ValueError(
            f"projection must be rank-2, got shape {tuple(projection.shape)}"
        )
    target_dim, source_dim = projection.shape

    resolved_config = target_config or ArrayProgramLibraryConfig(
        similarity_threshold=source.config.similarity_threshold,
        max_entries=source.config.max_entries,
        normalize_epsilon=source.config.normalize_epsilon,
    )
    target = ArrayProgramLibrary(resolved_config)

    projection_cpu = projection.detach().to(torch.float32).cpu()

    for entry in source.entries:
        if len(entry.signature) != source_dim:
            raise ValueError(
                f"entry signature dim {len(entry.signature)} does not match "
                f"projection source dim {source_dim}"
            )
        source_vec = torch.tensor(entry.signature, dtype=torch.float32)
        transferred = projection_cpu @ source_vec
        norm = float(torch.linalg.norm(transferred).item())
        if norm < resolved_config.normalize_epsilon:
            # Projection collapsed this signature — skip rather than record
            # a zero-norm entry, which `record()` would reject anyway.
            continue
        transferred_unit = (transferred / norm).tolist()
        new_entry = LibraryEntry(
            signature=[float(v) for v in transferred_unit],
            program=entry.program,
            hit_count=0,
            task_name=(
                (entry.task_name or "") + task_name_suffix
                if entry.task_name
                else None
            ),
            cached_at_step=entry.cached_at_step,
            convergence_gap=entry.convergence_gap,
        )
        target._entries.append(new_entry)

    # Post-transfer capacity cap (same rule as record()).
    if len(target._entries) > target.config.max_entries:
        target._entries.sort(key=lambda e: e.hit_count, reverse=True)
        del target._entries[target.config.max_entries :]
    return target


@dataclass
class ArrayThoughtLibraryResult:
    """Outcome of a library-aware forward pass."""

    predicted_output: torch.Tensor
    next_hidden_state: torch.Tensor
    programs: list[DiscreteArrayProgram] = field(default_factory=list)
    program_texts: list[str] = field(default_factory=list)
    library_hits: list[bool] = field(default_factory=list)
    newly_cached: list[bool] = field(default_factory=list)
    convergence_gaps: list[float] = field(default_factory=list)


__all__ = [
    "DiscreteArrayProgram",
    "ArrayProgramLibrary",
    "ArrayProgramLibraryConfig",
    "LibraryEntry",
    "ArrayThoughtLibraryResult",
    "transfer_library",
    "merge_libraries",
]
