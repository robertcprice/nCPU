"""Regression tests for ROADMAP Rung 9 Phase A (Program Prior Net tier-0).

Fast path (default): parses artifacts/prior_net_phase_a.json (produced by
training/prior_net/eval_phase_a.py; v1 schema {"v1": ..., "v0": ...}) and pins:
  - bench coverage 105/105 with the prior ON and OFF (DoD: coverage can
    never regress below the search baseline);
  - the v1 success criteria on the direct fallback head-to-head:
    16/16 both ways, >= 2 zero-search wins, ON wall <= OFF wall;
  - the v0 baseline history is preserved across artifact regenerations;
  - the calibrated gate (confidence_calibration.json) is consistent with
    the defaults compiled into prior_gen.rs.

Bridge path: round-trips a problem through nsynth/scripts/prior_net/propose.py
and checks the proposal JSON shape + gating behavior (skips if torch or the
model is missing).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = PROJECT_ROOT / "artifacts" / "prior_net_phase_a.json"
PROPOSE = PROJECT_ROOT / "nsynth" / "scripts" / "prior_net" / "propose.py"
CALIBRATION = PROJECT_ROOT / "training" / "prior_net" / "confidence_calibration.json"
PRIOR_GEN_RS = (
    PROJECT_ROOT / "nsynth" / "src" / "synthesis" / "universal_array" / "prior_gen.rs"
)
GEN_STATS = PROJECT_ROOT / "artifacts" / "prior_net_gen_stats.json"

N_ARR_SLOTS = 6
N_ARR_BODY = 4
SIGNALS = {"mean_max", "mean_margin", "mean_logp", "min_max"}


def _model() -> Path:
    """Same preference order as prior_gen.rs find_prior_net_assets."""
    for name in ("prior_net_v1.pt", "prior_net_v0.pt"):
        p = PROJECT_ROOT / "training" / "prior_net" / name
        if p.exists():
            return p
    pytest.skip("no prior-net checkpoint present")


def _artifact() -> dict:
    if not ARTIFACT.exists():
        pytest.skip(f"artifact missing: {ARTIFACT}")
    art = json.loads(ARTIFACT.read_text())
    if "v1" not in art:
        pytest.skip("artifact predates the v1 schema — rerun eval_phase_a.py")
    return art


def test_phase_a_coverage_pinned_105_both_runs():
    art = _artifact()["v1"]
    cov = art["coverage_both_runs"]
    assert cov["total"] == 105, f"bench size changed: {cov}"
    assert cov["off"] == 105, f"coverage regressed with prior OFF: {cov}"
    assert cov["on"] == 105, f"coverage regressed with prior ON: {cov}"


def test_phase_a_v0_history_preserved():
    art = _artifact()
    v0 = art["v0"]
    assert v0, "v0 baseline history dropped from the artifact"
    assert v0["coverage_both_runs"]["off"] == 105
    assert v0["direct_fallback"]["zero_search_solves"] == 2


def test_phase_a_v1_direct_fallback_success_criteria():
    """The Phase A v1 goal, pinned: flip v0's +33s to a net win while
    keeping >= 2 zero-search wins and full fallback coverage."""
    fb = _artifact()["v1"]["direct_fallback"]
    assert fb["problems"] == 16
    assert fb["solved"]["off"] == 16, "fallback coverage regressed (OFF)"
    assert fb["solved"]["on"] == 16, "fallback coverage regressed (ON)"
    assert fb["zero_search_solves"] >= 2, f"zero-search wins: {fb['zero_search_names']}"
    wall = fb["wall_seconds"]
    assert wall["on"] <= wall["off"], (
        f"prior ON must not cost wall time: OFF {wall['off']}s -> ON {wall['on']}s"
    )


def test_phase_a_artifact_schema_complete():
    art = _artifact()["v1"]
    for key in (
        "zero_search_solves",
        "prior_warm_solves",
        "wall_seconds",
        "method_counts",
        "per_problem",
        "isolation",
        "prior_config",
    ):
        assert key in art, f"artifact missing key {key}"
    assert isinstance(art["zero_search_solves"], int)
    assert len(art["per_problem"]) == 105


def test_phase_a_generator_stats_recorded():
    if not GEN_STATS.exists():
        pytest.skip(f"missing {GEN_STATS}")
    stats = json.loads(GEN_STATS.read_text())
    assert stats["written"] >= 100_000
    assert stats["distinct_codes"] > 1_000, "dataset collapsed to few programs"


def test_calibration_consistent_with_rust_defaults():
    """confidence_calibration.json's chosen tau/signal must match the
    DEFAULT_PRIOR_TAU / DEFAULT_PRIOR_SIGNAL compiled into prior_gen.rs —
    otherwise the deployed gate is not the calibrated gate."""
    if not CALIBRATION.exists():
        pytest.skip(f"missing {CALIBRATION}")
    cal = json.loads(CALIBRATION.read_text())
    assert cal["signal"] in SIGNALS
    tau = cal["chosen_tau"]
    assert tau is not None, "calibration found no usable tau"

    src = PRIOR_GEN_RS.read_text()
    m_tau = re.search(r"DEFAULT_PRIOR_TAU:\s*f64\s*=\s*([-+0-9.eE]+)\s*;", src)
    m_sig = re.search(r'DEFAULT_PRIOR_SIGNAL:\s*&str\s*=\s*"(\w+)"', src)
    assert m_tau and m_sig, "prior_gen.rs gate constants not found"
    assert abs(float(m_tau.group(1)) - float(tau)) < 1e-9, (
        f"Rust DEFAULT_PRIOR_TAU {m_tau.group(1)} != calibrated {tau}"
    )
    assert m_sig.group(1) == cal["signal"], (
        f"Rust DEFAULT_PRIOR_SIGNAL {m_sig.group(1)} != calibrated {cal['signal']}"
    )


def _sum_request() -> dict:
    return {
        "n_scalar": 0,
        "examples": [
            {"array": [1, 2, 3], "scalars": [], "expected": 6},
            {"array": [4, 5], "scalars": [], "expected": 9},
            {"array": [10], "scalars": [], "expected": 10},
            {"array": [-1, -2, -3, 4], "scalars": [], "expected": -2},
        ],
    }


def _run_propose(extra_args: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, str(PROPOSE), "--model", str(_model()), *extra_args],
        input=json.dumps(_sum_request()),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_propose_bridge_emits_valid_proposals():
    pytest.importorskip("torch")
    resp = _run_propose(["--k", "4"])
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


def test_propose_bridge_gates_below_tau():
    """tau above the signal ceiling must gate: no proposals, gated=true."""
    pytest.importorskip("torch")
    resp = _run_propose(["--tau", "1.1", "--signal", "mean_max"])
    assert resp["gated"] is True
    assert resp["proposals"] == []
    assert isinstance(resp["confidence"], float)


@pytest.mark.parametrize("signal", sorted(SIGNALS))
def test_propose_bridge_signals_emit_finite_confidence(signal):
    pytest.importorskip("torch")
    resp = _run_propose(["--tau", "-1e9", "--signal", signal])
    assert resp["gated"] is False
    assert len(resp["proposals"]) >= 1
    conf = resp["confidence"]
    assert isinstance(conf, float)
    if signal != "mean_logp":
        assert 0.0 <= conf <= 1.0
    else:
        assert conf <= 0.0
