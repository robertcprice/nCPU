"""Honest capability probe for TWO-ARGUMENT rules.

Real requirements are mostly multi-argument: `cost(base, units)`,
`ship(weight, zone)`, `bill(seats, overage)`. The 1-arg piecewise solver cannot
touch them. This measures what the engine does today on random 2-arg rules of two
realistic shapes — pure affine `c0*a + c1*b + c2`, and a single threshold on one
argument with an affine piece on each side — checking the synthesized program on
UNSEEN (a, b) pairs.

Usage: PYTHONPATH=. python3 demos/requirements_app/measure_generalization_2arg.py [N] [seed]
"""

from __future__ import annotations

import random
import sys

from ncpu.synthesis_api.server import SynthConfig, handle_synthesize_request


def make_2arg(rng: random.Random):
    """Random 2-arg rule. Half are pure affine, half a single threshold on a."""
    kind = rng.choice(["affine", "threshold"])
    c0 = rng.choice([0, 1, 2, 3, 5])
    c1 = rng.choice([0, 1, 2, 3, 5])
    c2 = rng.choice([0, 5, 10, -5])
    if kind == "affine":
        def fn(a: int, b: int) -> int:
            return c0 * a + c1 * b + c2
        return fn, f"affine c0={c0} c1={c1} c2={c2}"
    # threshold on a: below k, 0 contribution from a; above, slope c0
    k = rng.choice([10, 50, 100, 1000])
    d1 = rng.choice([1, 2, 3])

    def fn(a: int, b: int) -> int:
        over = a - k
        return (c0 * over if over > 0 else 0) + d1 * b + c2

    return fn, f"threshold k={k} c0={c0} d1={d1} c2={c2}"


def sample_2arg(fn, rng: random.Random):
    pts = set()
    for _ in range(14):
        a = rng.randint(0, 2000)
        b = rng.randint(0, 200)
        pts.add((a, b))
    # a few near small values
    for _ in range(4):
        pts.add((rng.randint(0, 200), rng.randint(0, 50)))
    rows = [((a, b), fn(a, b)) for a, b in sorted(pts)]
    return rows


def classify(fn, code_py: str, entry: str):
    if not code_py:
        return "FAILED"
    ns: dict = {}
    try:
        exec(code_py, ns)  # noqa: S102 — our own transpile, verification only
        g = ns[entry]
    except Exception:
        return "FAILED"
    bad = 0
    rng = random.Random(999)
    for _ in range(400):
        a = rng.randint(0, 3000)
        b = rng.randint(0, 300)
        try:
            if g(a, b) != fn(a, b):
                bad += 1
        except Exception:
            bad += 1
    return "SOLVED" if bad == 0 else "OVERFIT"


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    cfg = SynthConfig(timeout_s=20.0, max_timeout_s=20.0)
    rng = random.Random(seed)
    counts = {"SOLVED": 0, "OVERFIT": 0, "FAILED": 0}
    by_kind: dict[str, dict[str, int]] = {}
    for i in range(n):
        fn, desc = make_2arg(rng)
        kind = desc.split()[0]
        rows = sample_2arg(fn, rng)
        entry = f"r2_{i}"
        req = {
            "name": entry,
            "signature": f"fn {entry}(a: i64, b: i64) -> i64",
            "examples": [{"inputs": [a, b], "expected": y} for (a, b), y in rows],
        }
        sc, pl = handle_synthesize_request(req, cfg)
        code_py = (pl.get("transpiled") or {}).get("python", "") if sc == 200 else ""
        verdict = classify(fn, code_py, entry)
        counts[verdict] += 1
        by_kind.setdefault(kind, {"SOLVED": 0, "OVERFIT": 0, "FAILED": 0})[verdict] += 1
        print(f"  [{i:2}] {verdict:7} ({desc})  method={pl.get('method')}")
    tot = sum(counts.values())
    print("\n" + "=" * 56)
    print(f"RAW nsynth on {tot} unseen 2-arg rules (examples only):")
    print(f"  SOLVED  {counts['SOLVED']:3}  ({100*counts['SOLVED']//tot}%)")
    print(f"  OVERFIT {counts['OVERFIT']:3}")
    print(f"  FAILED  {counts['FAILED']:3}")
    for kind in sorted(by_kind):
        c = by_kind[kind]
        print(f"    {kind}: solved {c['SOLVED']}/{sum(c.values())}  overfit {c['OVERFIT']}  failed {c['FAILED']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
