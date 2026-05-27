"""Sanity checks for scripts/dev/clean_caches.sh.

The script is a dev utility, so we don't run its destructive paths here —
we just confirm the help surface and dry-run behavior stay stable so the
script doesn't silently start deleting the wrong things.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "dev" / "clean_caches.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_script_is_executable() -> None:
    assert SCRIPT.exists(), "clean_caches.sh missing"
    assert SCRIPT.stat().st_mode & 0o111, "clean_caches.sh must be executable"


def test_help_lists_safe_and_optional_targets() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout
    # Safe defaults that should always run
    assert "kernels/rust_metal/target" in out
    assert "__pycache__" in out
    assert ".pytest_cache" in out
    # Opt-in flags
    assert "--dry-run" in out
    assert "--prune-previews" in out
    assert "--pip-cache" in out
    assert "--cargo-registry" in out


def test_dry_run_does_not_delete_existing_cache(tmp_path: Path) -> None:
    # Create a fake __pycache__ under a scratch repo layout and make sure
    # --dry-run leaves it alone.
    repo = tmp_path / "fake_repo"
    (repo / "kernels" / "rust_metal").mkdir(parents=True)
    (repo / "pkg" / "__pycache__").mkdir(parents=True)
    sentinel = repo / "pkg" / "__pycache__" / "sentinel.pyc"
    sentinel.write_text("bytecode", encoding="utf-8")
    (repo / "kernels" / "rust_metal" / "target").mkdir()
    (repo / "kernels" / "rust_metal" / "target" / "debug.so").write_text(
        "fake", encoding="utf-8"
    )

    # Copy the script into the scratch repo to exercise its repo_root detection.
    scratch_script = repo / "scripts" / "dev" / "clean_caches.sh"
    scratch_script.parent.mkdir(parents=True)
    scratch_script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    scratch_script.chmod(0o755)

    subprocess.run(
        ["bash", str(scratch_script), "--dry-run"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )

    assert sentinel.exists(), "dry-run must not delete __pycache__"
    assert (repo / "kernels" / "rust_metal" / "target" / "debug.so").exists(), (
        "dry-run must not delete kernels/rust_metal/target contents"
    )


def test_real_run_removes_rust_target_and_pycache(tmp_path: Path) -> None:
    repo = tmp_path / "fake_repo"
    (repo / "kernels" / "rust_metal" / "target").mkdir(parents=True)
    (repo / "kernels" / "rust_metal" / "target" / "libthing.rlib").write_text(
        "x", encoding="utf-8"
    )
    (repo / "pkg" / "__pycache__").mkdir(parents=True)
    (repo / "pkg" / "__pycache__" / "stale.pyc").write_text("x", encoding="utf-8")
    (repo / "pkg" / "keep_me.py").write_text("print('hi')\n", encoding="utf-8")

    scratch_script = repo / "scripts" / "dev" / "clean_caches.sh"
    scratch_script.parent.mkdir(parents=True)
    scratch_script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    scratch_script.chmod(0o755)

    subprocess.run(
        ["bash", str(scratch_script)],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )

    assert not (repo / "kernels" / "rust_metal" / "target").exists(), (
        "real run must remove kernels/rust_metal/target"
    )
    assert not (repo / "pkg" / "__pycache__").exists(), (
        "real run must remove __pycache__ trees"
    )
    assert (repo / "pkg" / "keep_me.py").exists(), (
        "real run must not touch sibling source files"
    )
