"""Regression tests for ROADMAP Rung 9 Phase A (Program Prior Net tier-0).

Fast path (default): parses artifacts/prior_net_phase_a.json (produced by
training/prior_net/eval_phase_a.py) and pins:
  - bench coverage 105/105 with the prior ON and OFF (DoD: coverage can
    never regress below the search baseline);
  - artifact schema completeness (zero-search counts, wall deltas).

Bridge path: round-trips a problem through nsynth/scripts/prior_net/propose.py
and checks the proposal JSON shape (skips if torch or the model is missing).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = PROJECT_ROOT / "artifacts" / "prior_net_phase_a.json"
PROPOSE = PROJECT_ROOT / "nsynth" / "scripts" / "prior_net" / "propose.py"
MODEL = PROJECT_ROOT / "training" / "prior_net" / "prior_net_v0.pt"
GEN_STATS = PROJECT_ROOT / "artifacts" / "prior_net_gen_stats.json"

N_ARR_SLOTS = 6
N_ARR_BODY = 4


def _artifact() -> dict:
    if not ARTIFACT.exists():
        pytest.skip(f"artifact missing: {ARTIFACT}")
    return json.loads(ARTIFACT.read_text())


def test_phase_a_coverage_pinned_105_both_runs():
    art = _artifact()
    cov = art["coverage_both_runs"]
    assert cov["total"] == 105, f"bench size changed: {cov}"
    assert cov["off"] == 105, f"coverage regressed with prior OFF: {cov}"
    assert cov["on"] == 105, f"coverage regressed with prior ON: {cov}"


def test_phase_a_artifact_schema_complete():
    art = _artifact()
    for key in (
        "zero_search_solves",
        "prior_warm_solves",
        "wall_seconds",
        "method_counts",
        "per_problem",
        "isolation",
    ):
        assert key in art, f"artifact missing key {key}"
    assert isinstance(art["zero_search_solves"], int)
    assert len(art["per_problem"]) == 105


def test_phase_a_generator_stats_recorded():
    if not GEN_STATS.exists():
        pytest.skip(f"missing {GEN_STATS}")
    stats = json.loads(GEN_STATS.read_text())
    assert stats["written"] == 100_000
    assert stats["distinct_codes"] > 1_000, "dataset collapsed to few programs"


def test_propose_bridge_emits_valid_proposals():
    if not MODEL.exists():
        pytest.skip(f"model missing: {MODEL}")
    pytest.importorskip("torch")
    req = {
        "n_scalar": 0,
        "examples": [
            {"array": [1, 2, 3], "scalars": [], "expected": 6},
            {"array": [4, 5], "scalars": [], "expected": 9},
            {"array": [10], "scalars": [], "expected": 10},
            {"array": [-1, -2, -3, 4], "scalars": [], "expected": -2},
        ],
    }
    proc = subprocess.run(
        [sys.executable, str(PROPOSE), "--model", str(MODEL), "--k", "4"],
        input=json.dumps(req),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    resp = json.loads(proc.stdout)
    proposals = resp["proposals"]
    assert 1 <= len(proposals) <= 4, f"bad proposal count: {len(proposals)}"
    pool_n = 8 + 6 + 0 + 6  # n_scalar = 0
    lip_n = 1 + 6 + 0
    for p in proposals:
        assert p["consts"][:3] == [0, 1, -1]
        assert len(p["consts"]) == 6
        assert len(p["slots"]) == N_ARR_SLOTS
        for slot in p["slots"]:
            assert len(slot) == 7
            op, s1, s2, cmp_, gl, gr, el = slot
            assert 0 <= op <= 5
            assert 0 <= cmp_ < 6
            for idx in (s1, s2, gl, gr, el):
                assert 0 <= idx < pool_n, f"pool index {idx} out of range"
        assert len(p["body_init"]) == N_ARR_BODY
        assert all(0 <= b < lip_n for b in p["body_init"])
        assert 0 <= p["ret"] < pool_n
