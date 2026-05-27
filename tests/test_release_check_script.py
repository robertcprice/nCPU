"""Pin the release_check.sh strict Rust/Metal smoke invocation.

The smoke exists so a broken Rust/Metal hotloop path fails the maintainer
release check before a full publication sweep. Small drift in the flags
(e.g. losing ``--rust-only``) would turn the smoke into a permissive run
silently, so we keep the critical contract parts pinned here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RELEASE_CHECK = PROJECT_ROOT / "scripts" / "release" / "release_check.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    return RELEASE_CHECK.read_text(encoding="utf-8")


def _extract_smoke_block(text: str) -> str:
    match = re.search(
        r"Strict Rust/Metal GPU-only smoke.*?Skipping strict Rust/Metal smoke",
        text,
        flags=re.DOTALL,
    )
    assert match is not None, "Strict Rust/Metal smoke block not found in release_check.sh"
    return match.group(0)


def test_smoke_invokes_export_matrix_with_strict_rust_contract(script_text: str) -> None:
    block = _extract_smoke_block(script_text)
    assert "benchmarks/export_gpu_only_matrix.py" in block
    assert "--rust-only" in block
    assert "--workload counted" in block
    assert "--timeout-seconds 60" in block
    assert "--output-dir" in block


def test_smoke_guarded_by_darwin_and_cargo(script_text: str) -> None:
    assert 'command -v cargo' in script_text
    assert '"$(uname -s)" == "Darwin"' in script_text
    assert "load_ncpu_metal" in script_text, (
        "smoke must probe the Rust/Metal extension via Python import, "
        "not via a hard-coded .abi3.so path"
    )


def test_smoke_failure_propagates(script_text: str) -> None:
    block = _extract_smoke_block(script_text)
    assert "exit 1" in block, "smoke block must fail-close on exporter non-zero exit"
