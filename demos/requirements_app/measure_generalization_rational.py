"""Honest capability probe for RATIONAL FLOOR-DIVISION (search_rational_floor).

Recovers `f(x) = (a*x + b) / d` (integer/floor) — the affine lives INSIDE the
division, which no affine or composition solver can express. Random non-affine
floor rules (a,b >= 0, d >= 2), examples ONLY, SOLVED only when exactly correct
on UNSEEN points. Training spans a wide x-range so the step structure of the true
divisor is observable (otherwise a coarser composed `x / m` fit is consistent
with the samples and is the honest answer given the data).

Usage:  PYTHONPATH=. python3 demos/requirements_app/measure_generalization_rational.py [N] [seed]
"""

from __future__ import annotations

import random
import sys

from ncpu.synthesis_api.server import SynthConfig, handle_synthesize_request


def make_rule(rng: random.Random):
    while True:
        d = rng.randint(2, 9)
        a = rng.randint(1, 12)
        b = rng.randint(0, 15)

        def fn(x, a=a, b=b, d=d):
            return (a * x + b) // d

        # Non-affine only (first differences must vary) — a constant-difference
        # rule is a plain affine, owned by search_affine.
        if len({fn(x + 1) - fn(x) for x in range(0, 30)}) > 1:
            return fn, f"({a}x+{b})/{d}"


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    cfg = SynthConfig(timeout_s=10.0, max_timeout_s=10.0)
    rng = random.Random(seed)
    counts = {"SOLVED": 0, "OVERFIT": 0, "FAILED": 0}
    via = 0
    misses = []

    for _ in range(n):
        fn, desc = make_rule(rng)
        train = [([x], fn(x)) for x in range(0, 45)]  # wide span → structure observable
        req = {
            "name": "rule",
            "signature": "fn rule(x0: i64) -> i64",
            "examples": [{"inputs": xs, "expected": y} for xs, y in train],
        }
        sc, pl = handle_synthesize_request(req, cfg)
        code_py = (pl.get("transpiled") or {}).get("python", "") if sc == 200 else ""
        verdict = "FAILED"
        bad = None
        if code_py:
            ns: dict = {}
            try:
                exec(code_py, ns)  # noqa: S102
                g = ns["rule"]
                bad = sum(1 for x in range(0, 70) if g(x) != fn(x))
                verdict = "SOLVED" if bad == 0 else "OVERFIT"
            except Exception:
                verdict = "FAILED"
        counts[verdict] += 1
        if pl.get("method") == "search_rational_floor":
            via += 1
        if verdict != "SOLVED" and len(misses) < 10:
            misses.append((verdict, desc, pl.get("method"), bad))

    print("=" * 60)
    tot = sum(counts.values())
    print(f"RAW nsynth on {tot} unseen rational-floor rules (examples only):")
    print(f"  SOLVED  (correct on unseen): {counts['SOLVED']:3}  ({100*counts['SOLVED']/tot:.0f}%)")
    print(f"  OVERFIT (wrong on unseen):   {counts['OVERFIT']:3}  ({100*counts['OVERFIT']/tot:.0f}%)")
    print(f"  FAILED  (no program):        {counts['FAILED']:3}  ({100*counts['FAILED']/tot:.0f}%)")
    print(f"  solved via search_rational_floor: {via}")
    for verdict, desc, method, bad in misses:
        print(f"    {verdict}: {desc}  via {method}  ({bad} mismatches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
