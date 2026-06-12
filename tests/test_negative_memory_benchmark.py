"""Validate the negative-memory benchmark artifact (REPRODUCIBILITY Claim 7).

The fast tests read the committed artifact at
``artifacts/negative_memory_benchmark.json`` (produced by
``benchmarks/benchmark_negative_memory.py``) and pin its shape and the
convergence result — they do NOT rerun the solver. The full end-to-end
cold/warm rerun (two cascade-exhaustion solver runs, ~5 min total) is
opt-in behind ``NCPU_NEGMEM_FULL=1``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "artifacts" / "negative_memory_benchmark.json"
HARNESS = REPO_ROOT / "benchmarks" / "benchmark_negative_memory.py"
BACKEND = REPO_ROOT / "nsynth" / "target" / "release" / "mog_synth"

REQUIRED_KEYS = {
    "problem": dict,
    "cold_rejections": int,
    "warm_new_rejections": int,
    "warm_total_rejections": int,
    "converged": bool,
    "wall_cold_s": (int, float),
    "wall_warm_s": (int, float),
}


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.is_file(), (
        f"missing committed artifact {ARTIFACT}; regenerate with: "
        f"python {HARNESS.relative_to(REPO_ROOT)}"
    )
    return json.loads(ARTIFACT.read_text())


def test_artifact_shape(artifact):
    for key, types in REQUIRED_KEYS.items():
        assert key in artifact, f"artifact missing key {key!r}"
        assert isinstance(artifact[key], types), (
            f"artifact[{key!r}] is {type(artifact[key]).__name__}, expected {types}"
        )


def test_artifact_converged(artifact):
    """The claim itself: the warm rerun added zero new rejections."""
    assert artifact["converged"] is True
    assert artifact["warm_new_rejections"] == 0
    assert artifact["warm_total_rejections"] == artifact["cold_rejections"]


def test_artifact_cold_run_did_real_work(artifact):
    """A refusal that never reached the candidate search proves nothing."""
    assert artifact["cold_rejections"] > 0
    assert artifact["wall_cold_s"] > 0
    assert artifact["wall_warm_s"] > 0


def test_artifact_problem_recorded(artifact):
    """The exact novel problem is embedded for third-party reproduction."""
    problem = artifact["problem"]
    assert isinstance(problem.get("name"), str) and problem["name"]
    examples = problem.get("examples")
    assert isinstance(examples, list) and len(examples) >= 3
    for example in examples:
        assert "inputs" in example and "expected" in example


@pytest.mark.skipif(
    os.environ.get("NCPU_NEGMEM_FULL") != "1",
    reason="full cold/warm solver rerun (~5 min); set NCPU_NEGMEM_FULL=1 to run",
)
def test_full_cold_warm_rerun_converges(tmp_path):
    if not BACKEND.is_file():
        pytest.skip(f"mog_synth release binary not built: {BACKEND}")
    out = tmp_path / "negative_memory_benchmark.json"
    proc = subprocess.run(
        [sys.executable, str(HARNESS), "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"harness failed ({proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    fresh = json.loads(out.read_text())
    assert fresh["converged"] is True
    assert fresh["warm_new_rejections"] == 0
    assert fresh["cold_rejections"] > 0
