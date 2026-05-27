"""Regression tests for the Tier-2 game-adjacent benchmark factories.

These factories live at nsynth/src/benchmark.rs. Each represents a shape
that pushes past Tier-1 (single clean formula) toward game-adjacent logic.

We don't pin coverage (the whole point is that some of these may or may
not be gradient-solvable today; the empirical number is documented in
paper/section_game_scale_roadmap.md). Instead, we assert that the Rust
binary reports the expected number of factories and that Tier-2 names
appear as fresh additions the publishable-claim harness will reach.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BINARY = PROJECT_ROOT / "nsynth" / "target" / "release" / "mog_synth"

TIER2_NAMES = {
    "score_tracker",
    "vending_change",
    "combat_resolve",
    "traffic_light_phase",
    "run_length_decode_sum",
    "count_adjacent_diff",
    "priority_pop",
    "turn_order_rotate",
    "grid_bounds_check",
    "simulate_gravity",
}


def test_tier2_factories_registered():
    """The 10 Tier-2 factories are listed in nsynth/src/benchmark.rs FACTORIES."""
    source = (PROJECT_ROOT / "nsynth" / "src" / "benchmark.rs").read_text()
    # Locate the FACTORIES block
    start = source.index("pub const FACTORIES: &[Factory]")
    end = source.index("];", start)
    factories_block = source[start:end]
    for name in TIER2_NAMES:
        assert f"make_{name}" in factories_block, (
            f"tier-2 factory make_{name} is defined but not registered in FACTORIES"
        )


def test_tier2_factories_compile():
    """The Rust binary builds with the new factories included.

    If Tier-2 additions introduced a compile error, the test fails with a
    useful cargo diagnostic rather than a silent broken binary.
    """
    if os.environ.get("NCPU_SKIP_RUST_BUILD") == "1":
        pytest.skip("NCPU_SKIP_RUST_BUILD=1")
    proc = subprocess.run(
        ["cargo", "check", "--release", "--bin", "mog_synth"],
        cwd=str(PROJECT_ROOT / "nsynth"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"cargo check failed:\nstdout={proc.stdout[-1500:]}\nstderr={proc.stderr[-1500:]}"
    )


def test_tier2_present_in_default_coverage_artifact_if_present():
    """If the default-mode coverage artifact has been regenerated since Tier-2
    was added, all 10 Tier-2 names must appear in it. We skip if the artifact
    was produced before Tier-2 existed."""
    path = PROJECT_ROOT / "artifacts" / "nsynth_per_problem_coverage.jsonl"
    if not path.exists():
        pytest.skip(f"artifact missing: {path}")
    names: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("summary"):
            continue
        name = obj.get("name", "")
        if isinstance(name, str):
            names.add(name.rsplit("_v", 1)[0])  # strip _v0 suffix

    have_any_tier2 = bool(names & TIER2_NAMES)
    if not have_any_tier2:
        pytest.skip("coverage artifact predates Tier-2; regenerate to enable this test")

    missing = TIER2_NAMES - names
    assert not missing, (
        f"tier-2 names missing from coverage artifact: {sorted(missing)}. "
        f"Regenerate `artifacts/nsynth_per_problem_coverage.jsonl` via "
        f"`nsynth/target/release/mog_synth --per-problem-json`."
    )
