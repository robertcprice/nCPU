"""Reproducibility and provenance helpers for benchmark/artifact generation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


def _run_text(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_snapshot(repo_root: Path) -> dict[str, Any]:
    return {
        "root": str(repo_root),
        "commit": _run_text(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "short_commit": _run_text(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root),
        "branch": _run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root),
        "describe": _run_text(["git", "describe", "--always", "--dirty", "--tags"], cwd=repo_root),
        "dirty": bool(_run_text(["git", "status", "--short"], cwd=repo_root)),
    }


def detect_mog_snapshot(project_root: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        "root": None,
        "commit": None,
        "short_commit": None,
        "toolchain": {},
    }

    try:
        from egdc.mog.execute import MOGC_BINARY, MOG_RUNTIME

        data["toolchain"] = {
            "mogc_binary": str(MOGC_BINARY),
            "mogc_exists": MOGC_BINARY.is_file(),
            "runtime_binary": str(MOG_RUNTIME),
            "runtime_exists": MOG_RUNTIME.is_file(),
        }
        for root in (
            Path(os.environ.get("MOG_ROOT", "")).expanduser() if os.environ.get("MOG_ROOT") else None,
            project_root.parent / "mog",
            Path.home() / "projects" / "mog",
        ):
            if root is None:
                continue
            if (root / ".git").exists():
                snap = git_snapshot(root)
                data["root"] = snap["root"]
                data["commit"] = snap["commit"]
                data["short_commit"] = snap["short_commit"]
                data["git"] = snap
                break
    except Exception as exc:
        data["error"] = str(exc)

    return data


def collect_provenance(project_root: Path, *, argv: list[str] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    project_root = project_root.resolve()
    provenance: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv if argv is None else argv),
        "cwd": str(Path.cwd()),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "dependencies": {
            name: package_version(name)
            for name in ("torch", "numpy", "scipy", "transformers", "peft")
        },
        "ncpu_git": git_snapshot(project_root),
        "mog": detect_mog_snapshot(project_root),
        "environment": {
            key: os.environ.get(key)
            for key in ("MOG_ROOT", "MOG_GIT_REF", "MOGC_BINARY", "MOG_RUNTIME")
            if os.environ.get(key)
        },
    }
    if extra:
        provenance["extra"] = extra
    return provenance


def file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    payload = resolved.read_bytes()
    record = {
        "path": str(resolved if root is None else resolved.relative_to(root.resolve())),
        "size_bytes": resolved.stat().st_size,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return record
