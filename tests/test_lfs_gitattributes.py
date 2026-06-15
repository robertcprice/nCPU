"""LFS-coverage guard for the nsynth data directory.

The committed `.gitattributes` lists every JSONL in `nsynth/data` larger
than 50 MB under LFS. This test re-scans the directory and fails if any
file slips through the rule — a missing entry would silently balloon the
git checkout. Small files (< 50 MB) are allowed in plain git; the rule
matches the Rung 6 hygiene plan.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "nsynth" / "data"
GITATTRIBUTES = REPO_ROOT / ".gitattributes"

LFS_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB


def _read_lfs_rules() -> set[str]:
    if not GITATTRIBUTES.is_file():
        return set()
    rules: set[str] = set()
    pattern = re.compile(r"^(\S+)\s+filter=lfs")
    for line in GITATTRIBUTES.read_text().splitlines():
        m = pattern.match(line)
        if m:
            rules.add(m.group(1).lstrip("/"))
    return rules


@pytest.mark.skipif(not DATA_DIR.is_dir(), reason="nsynth/data not present")
def test_large_jsonls_are_tracked_via_lfs():
    rules = _read_lfs_rules()
    offenders: list[tuple[Path, int]] = []
    for path in DATA_DIR.glob("*.jsonl"):
        size = path.stat().st_size
        if size < LFS_THRESHOLD_BYTES:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel not in rules:
            offenders.append((path, size))
    assert not offenders, (
        "These JSONLs exceed the LFS threshold but are not covered by "
        ".gitattributes:\n  " + "\n  ".join(
            f"{p} ({n / 1024 / 1024:.1f} MB)" for p, n in offenders
        )
    )


def test_gitattributes_is_parsed():
    # Sanity: the parser handles the existing file.
    rules = _read_lfs_rules()
    assert "*.pt" in rules
    assert "nsynth/data/prior_net_train_300k.jsonl" in rules
