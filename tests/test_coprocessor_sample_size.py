"""Sample-size regressions for coprocessor real-world evaluation results.

Enforces the statistical protocol described in paper/section_sample_size.md:
any real-world benchmark JSON whose deltas appear in paper/ must be derived
from a sample of N >= 100 per benchmark. Runs on committed artifacts when
they exist; skips gracefully when they don't so the test is safe to ship
before the vast.ai results land.

The test is structural (checks the N, not the accuracy) to avoid pinning a
specific coprocessor delta that's expected to shift as we learn more.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTRUCT_SWEEP = PROJECT_ROOT / "training_results" / "instruct_sweep"

# Canonical artifacts produced by the vast.ai deploy. Each represents one model.
REALWORLD_ARTIFACTS = [
    INSTRUCT_SWEEP / "qwen3.5-4b" / "realworld_vastai.json",
    INSTRUCT_SWEEP / "qwen3.5-9b" / "realworld_vastai.json",
]

# Minimum N enforced by the sample-size protocol (see paper/section_sample_size.md §18.2).
MIN_N_PER_BENCHMARK = 100


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.mark.parametrize("path", REALWORLD_ARTIFACTS)
def test_realworld_artifact_meets_minimum_n(path: Path):
    if not path.exists():
        pytest.skip(f"artifact not yet produced: {path}")
    data = _load(path)
    # The benchmark JSON has a top-level "baseline" and "coprocessor" keyed
    # by benchmark name, each with 'total'. Enforce N >= 100 per benchmark.
    for layer in ("baseline", "coprocessor"):
        assert layer in data, f"missing {layer} in {path.name}"
        for bench_name, bench in data[layer].items():
            total = bench.get("total")
            assert isinstance(total, int), (
                f"{path.name} {layer}.{bench_name}.total is not an int: {total!r}"
            )
            assert total >= MIN_N_PER_BENCHMARK, (
                f"{path.name} {layer}.{bench_name} N={total} < {MIN_N_PER_BENCHMARK} "
                f"— see paper/section_sample_size.md"
            )


@pytest.mark.parametrize("path", REALWORLD_ARTIFACTS)
def test_realworld_artifact_baseline_and_coprocessor_have_same_n(path: Path):
    if not path.exists():
        pytest.skip(f"artifact not yet produced: {path}")
    data = _load(path)
    baseline = data.get("baseline", {})
    coproc = data.get("coprocessor", {})
    # Each benchmark name should appear in both layers at the same N,
    # otherwise the delta is meaningless.
    shared = set(baseline.keys()) & set(coproc.keys())
    for name in shared:
        b_n = baseline[name].get("total")
        c_n = coproc[name].get("total")
        assert b_n == c_n, (
            f"{path.name} benchmark '{name}' has baseline N={b_n} "
            f"but coprocessor N={c_n} — deltas require matched N"
        )


@pytest.mark.parametrize("path", REALWORLD_ARTIFACTS)
def test_realworld_artifact_records_model_identity(path: Path):
    if not path.exists():
        pytest.skip(f"artifact not yet produced: {path}")
    data = _load(path)
    # The artifact must record which HF model was evaluated. Otherwise the
    # per-model numbers are untraceable.
    assert data.get("model"), f"{path.name} missing 'model' identity"
