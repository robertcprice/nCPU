"""Honest capability probe for MULTI-ARGUMENT TIERED rules — affine in two
arguments within each tier of one of them (e.g. a shipping cost linear in weight
and zone, tiered by weight). Examples only; correctness checked on unseen pairs.

Usage: PYTHONPATH=. python3 demos/requirements_app/measure_generalization_multiarg_tiered.py [N] [seed]
"""

from __future__ import annotations

import random
import sys

from ncpu.synthesis_api.server import SynthConfig, handle_synthesize_request


def make_rule(rng: random.Random):
    """Continuous tiered-in-`a`, affine-in-(a,b). 2–3 tiers on `a`, slope on b
    constant across tiers (a realistic 'per-unit-b plus tiered-a' schedule)."""
    n_tiers = rng.randint(2, 3)
    bps = sorted(rng.sample(range(50, 1800), n_tiers - 1))
    slopes_a = [rng.choice([0, 1, 2, 3, 5]) for _ in range(n_tiers)]
    slope_b = rng.choice([0, 1, 2, 3])
    base = rng.choice([0, 5, 10])

    def fn(a: int, b: int) -> int:
        edges = [0] + bps
        acc = base
        for i in range(n_tiers):
            lo = edges[i]
            hi = bps[i] if i < len(bps) else None
            if hi is not None and a > hi:
                acc += slopes_a[i] * (hi - lo)
            elif a > lo:
                acc += slopes_a[i] * (a - lo)
                break
            else:
                break
        return acc + slope_b * b

    return fn, f"{n_tiers}tier bps={bps} sa={slopes_a} sb={slope_b}"


def sample(fn, rng: random.Random):
    hi = 2500
    pts = set()
    for _ in range(20):
        pts.add((rng.randint(0, hi), rng.randint(0, 100)))
    for _ in range(4):
        pts.add((rng.randint(0, 200), rng.randint(0, 50)))
    return [((a, b), fn(a, b)) for a, b in sorted(pts)]


def classify(fn, code_py: str, entry: str):
    if not code_py:
        return "FAILED"
    ns: dict = {}
    try:
        exec(code_py, ns)  # noqa: S102
        g = ns[entry]
    except Exception:
        return "FAILED"
    rng = random.Random(123)
    for _ in range(500):
        a, b = rng.randint(0, 3000), rng.randint(0, 150)
        try:
            if g(a, b) != fn(a, b):
                return "OVERFIT"
        except Exception:
            return "OVERFIT"
    return "SOLVED"


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    cfg = SynthConfig(timeout_s=15.0, max_timeout_s=15.0)
    rng = random.Random(seed)
    counts = {"SOLVED": 0, "OVERFIT": 0, "FAILED": 0}
    for i in range(n):
        fn, desc = make_rule(rng)
        rows = sample(fn, rng)
        entry = f"mt_{i}"
        req = {
            "name": entry,
            "signature": f"fn {entry}(a: i64, b: i64) -> i64",
            "examples": [{"inputs": [a, b], "expected": y} for (a, b), y in rows],
        }
        sc, pl = handle_synthesize_request(req, cfg)
        code_py = (pl.get("transpiled") or {}).get("python", "") if sc == 200 else ""
        v = classify(fn, code_py, entry)
        counts[v] += 1
        print(f"  [{i:2}] {v:7} ({desc}) method={pl.get('method')}")
    tot = sum(counts.values())
    print("\n" + "=" * 56)
    print(f"MULTI-ARG TIERED, {tot} unseen rules (examples only):")
    print(f"  SOLVED  {counts['SOLVED']:3} ({100*counts['SOLVED']//tot}%)")
    print(f"  OVERFIT {counts['OVERFIT']:3}")
    print(f"  FAILED  {counts['FAILED']:3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
