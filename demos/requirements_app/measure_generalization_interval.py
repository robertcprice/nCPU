"""Honest capability probe for INTERVAL-MEMBERSHIP branches (search_interval_branch).

Recovers `if lo <= x && x <= hi { A(x) } else { B(x) }` — the first solver to emit
the logical `&&` operator. Single-argument closed intervals, recovered by
deterministic 3-run segmentation. Random interval rules, examples ONLY, SOLVED
only when exactly correct on UNSEEN points. Training covers below / inside /
above the band so the interval is observable (the fair test).

Usage:  PYTHONPATH=. python3 demos/requirements_app/measure_generalization_interval.py [N] [seed]
"""

from __future__ import annotations

import random
import sys

from ncpu.synthesis_api.server import SynthConfig, handle_synthesize_request


def make_interval(rng: random.Random):
    lo = rng.randint(5, 18)
    hi = lo + rng.randint(4, 14)
    a0, a1 = rng.choice([0, 1, 3, 5, -2]), rng.choice([1, 2, 3, -1, -2])
    b0, b1 = rng.choice([0, 1, 3, 5, -2]), rng.choice([1, 2, 3, -1, -2])
    while (a0, a1) == (b0, b1):
        b0, b1 = rng.choice([0, 1, 3, 5, -2]), rng.choice([1, 2, 3, -1, -2])

    def fn(x):
        return (a0 + a1 * x) if lo <= x <= hi else (b0 + b1 * x)

    return fn, f"if {lo}<=x<={hi}: {a0}+{a1}x else {b0}+{b1}x", lo, hi


def sample_training(fn, lo, hi):
    """Dense coverage of all three regions — below lo, inside [lo,hi], above hi —
    so both boundaries are pinned and the interval is observable."""
    xs = sorted(set(list(range(0, lo)) + list(range(lo, hi + 1)) + list(range(hi + 1, hi + 11))))
    return [([x], fn(x)) for x in xs]


def classify(fn, code_py, hi):
    if not code_py:
        return "FAILED", None
    ns: dict = {}
    try:
        exec(code_py, ns)  # noqa: S102
        g = ns["rule"]
    except Exception:
        return "FAILED", None
    bad = sum(1 for x in range(0, hi + 13) if g(x) != fn(x))
    return ("SOLVED" if bad == 0 else "OVERFIT"), bad


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    cfg = SynthConfig(timeout_s=10.0, max_timeout_s=10.0)
    rng = random.Random(seed)
    counts = {"SOLVED": 0, "OVERFIT": 0, "FAILED": 0}
    via = 0
    misses = []

    for _ in range(n):
        fn, desc, lo, hi = make_interval(rng)
        train = sample_training(fn, lo, hi)
        req = {
            "name": "rule",
            "signature": "fn rule(x0: i64) -> i64",
            "examples": [{"inputs": xs, "expected": y} for xs, y in train],
        }
        sc, pl = handle_synthesize_request(req, cfg)
        code_py = (pl.get("transpiled") or {}).get("python", "") if sc == 200 else ""
        verdict, bad = classify(fn, code_py, hi)
        counts[verdict] += 1
        if pl.get("method") == "search_interval_branch":
            via += 1
        if verdict != "SOLVED" and len(misses) < 10:
            misses.append((verdict, desc, pl.get("method"), bad))

    print("=" * 60)
    tot = sum(counts.values())
    print(f"RAW nsynth on {tot} unseen interval-branch rules (examples only):")
    print(f"  SOLVED  (correct on unseen): {counts['SOLVED']:3}  ({100*counts['SOLVED']/tot:.0f}%)")
    print(f"  OVERFIT (wrong on unseen):   {counts['OVERFIT']:3}  ({100*counts['OVERFIT']/tot:.0f}%)")
    print(f"  FAILED  (no program):        {counts['FAILED']:3}  ({100*counts['FAILED']/tot:.0f}%)")
    print(f"  solved via search_interval_branch: {via}")
    for verdict, desc, method, bad in misses:
        print(f"    {verdict}: {desc}  via {method}  ({bad} mismatches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
