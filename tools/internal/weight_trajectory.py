#!/usr/bin/env python3
"""
Analyze the weight-history TSV and report per-dimension drift over time.

Reads artifacts/meta_weights_history.tsv (one row per `weights_snapshot`
call: timestamp + 26 weights + optional label) and emits:
  - Per-dimension min/max/final values
  - Dimensions that drifted most from the uniform-1.0 prior
  - A compact sparkline per dimension so a reader can see the trajectory
    at a glance without external plotting

Pure Python 3 standard library. No numpy, no pandas — works on any CI
image with Python installed. Designed to be cron-able alongside the
weekly self-improvement measurement.

Usage:
    python3 tools/weight_trajectory.py \
        [--in artifacts/meta_weights_history.tsv] \
        [--out artifacts/WEIGHT_TRAJECTORY.md]
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional

PRIOR = 1.0
# Unicode block sparkline. 8 levels; we map weights [0, 2] to these blocks.
SPARKS = " ▁▂▃▄▅▆▇█"

# Human-readable names for each weight dimension. Keep in sync with the
# FEATURE_DIM doc comment in nsynth/src/meta_learner.rs. Using a dict (not a
# list) so missing/extra dimensions in a weight file don't crash the report.
FEATURE_NAMES: dict[int, str] = {
    0: "n_args",
    1: "n_examples",
    2: "output spread (max-min)",
    3: "mean output",
    4: "mean |output|",
    5: "monotone-in-arg0 score",
    6: "mean out / mean arg0",
    7: "fraction outputs ≥ 0",
    8: "code length (bytes)",
    9: "code has '*'",
    10: "code has '+'",
    11: "code has '-'",
    12: "code has '%'",
    13: "code has '/'",
    14: "code has 'if'",
    15: "code has 'for'/'while'",
    16: "code '{' depth proxy",
    17: "code has 'return'",
    18: "log1p('+' count)",
    19: "log1p('-' count)",
    20: "log1p('*' count)",
    21: "log1p('/' count)",
    22: "log1p('%' count)",
    23: "log1p('if ' count)",
    24: "log1p(loop-keyword count)",
    25: "log1p('return' count)",
    26: "n_args × monotone",
    27: "mean_abs_out × (1-monotone)",
    28: "n_examples × n_args",
    29: "ratio × fraction_nonneg",
    30: "has_loop × has_branch",
    31: "has_mul × has_mod",
}


def feature_name(dim: int) -> str:
    return FEATURE_NAMES.get(dim, f"(dim {dim})")


@dataclass
class WeightRow:
    ts: int
    weights: List[float]
    label: str


def load_rows(path: str) -> List[WeightRow]:
    """Parse the TSV. Tolerates trailing label columns."""
    rows: List[WeightRow] = []
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if not parts:
                continue
            try:
                ts = int(parts[0])
            except ValueError:
                continue
            weights: List[float] = []
            label = ""
            for tok in parts[1:]:
                try:
                    weights.append(float(tok))
                except ValueError:
                    # Anything that didn't parse as float is a trailing label.
                    label = tok
                    break
            if weights:
                rows.append(WeightRow(ts=ts, weights=weights, label=label))
    rows.sort(key=lambda r: r.ts)
    return rows


def sparkline(values: List[float], lo: float = 0.0, hi: float = 2.0) -> str:
    """Render a list of floats to an 8-level block sparkline.
    Values outside [lo, hi] are clamped; keeps the visualization stable
    when a single outlier would otherwise compress everything."""
    if not values:
        return ""
    out = []
    span = max(hi - lo, 1e-9)
    for v in values:
        clamped = max(lo, min(hi, v))
        level = int(round((clamped - lo) / span * (len(SPARKS) - 1)))
        out.append(SPARKS[level])
    return "".join(out)


def summarise(rows: List[WeightRow]) -> str:
    if not rows:
        return "No weight snapshots in history — run `weights_snapshot` first.\n"

    n_dims = max(len(r.weights) for r in rows)
    # Pad rows to consistent length in case an older snapshot had fewer dims.
    for r in rows:
        if len(r.weights) < n_dims:
            r.weights.extend([PRIOR] * (n_dims - len(r.weights)))

    lines: List[str] = []
    lines.append("# Weight Trajectory")
    lines.append("")
    lines.append(f"{len(rows)} snapshot(s), {n_dims} dimensions.")
    lines.append(f"First: {rows[0].ts}  Last: {rows[-1].ts}  "
                 f"Span: {rows[-1].ts - rows[0].ts} s")
    lines.append("")
    lines.append("Per-dimension drift from the uniform prior (1.0). Sparkline")
    lines.append("shows trajectory across the recorded snapshots (ascending time).")
    lines.append("")
    lines.append(
        "| dim | feature | min | max | final | Δ from prior | trajectory |"
    )
    lines.append(
        "|----:|:--------|----:|----:|------:|-------------:|:-----------|"
    )

    for i in range(n_dims):
        series = [r.weights[i] for r in rows]
        mn, mx = min(series), max(series)
        final = series[-1]
        delta = final - PRIOR
        spark = sparkline(series)
        lines.append(
            f"| {i} | {feature_name(i)} | {mn:.3f} | {mx:.3f} | {final:.3f} | {delta:+.3f} | `{spark}` |"
        )

    # Top drifters — dimensions with the largest absolute delta.
    lines.append("")
    lines.append("## Top drifters (largest |final − prior|)")
    lines.append("")
    drifts = sorted(
        [(i, rows[-1].weights[i] - PRIOR) for i in range(n_dims)],
        key=lambda p: abs(p[1]),
        reverse=True,
    )
    for dim, delta in drifts[:8]:
        direction = "↑" if delta > 0 else "↓"
        lines.append(
            f"- dim {dim} ({feature_name(dim)}): {rows[-1].weights[dim]:.4f} "
            f"({direction} {abs(delta):+.4f} from prior)"
        )

    # Stability score: mean absolute per-step change across all dimensions.
    # Low values mean the weight space has settled; high values mean
    # learning is still actively pulling weights around.
    if len(rows) >= 2:
        per_step = []
        for i in range(n_dims):
            for j in range(1, len(rows)):
                per_step.append(abs(rows[j].weights[i] - rows[j - 1].weights[i]))
        if per_step:
            stab = sum(per_step) / len(per_step)
            lines.append("")
            lines.append(f"## Stability")
            lines.append("")
            lines.append(
                f"Mean per-step |Δw| = **{stab:.5f}**. Lower means the weight"
                f" space is settling; higher means learning is still moving weights."
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--in",
        dest="in_path",
        default="artifacts/meta_weights_history.tsv",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        default="artifacts/WEIGHT_TRAJECTORY.md",
    )
    args = ap.parse_args()

    if not os.path.exists(args.in_path):
        raise SystemExit(
            f"[weight_trajectory] {args.in_path} not found — "
            f"run `cargo run --release --bin weights_snapshot` first"
        )

    rows = load_rows(args.in_path)
    report = summarise(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)
    with open(args.out_path, "w") as f:
        f.write(report)

    print(f"[weight_trajectory] wrote {args.out_path}")
    # Also print a quick one-liner summary.
    if rows:
        total_abs_delta = sum(abs(w - PRIOR) for w in rows[-1].weights)
        print(
            f"[weight_trajectory] {len(rows)} snapshots, "
            f"total |Δ| from prior = {total_abs_delta:.4f}"
        )


if __name__ == "__main__":
    main()
