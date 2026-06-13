"""Honest capability probe: can nsynth, given ONLY examples (no reference, no
CEGIS, no curation), synthesize piecewise/tiered rules that GENERALIZE to unseen
inputs?

For each randomly-generated continuous piecewise-affine function (the shape of
real tiered pricing) we:
  1. sample a modest set of training examples spanning the range,
  2. call the raw synthesizer on examples only,
  3. evaluate the returned program on DENSE UNSEEN points against ground truth.

A program is SOLVED only if it is exactly correct on every unseen point; if it
synthesizes something that fits the training samples but is wrong on unseen
points it is an OVERFIT (the failure mode that matters). This is the real
capability number — nothing here is hand-picked to pass.

Usage:  PYTHONPATH=. python3 demos/requirements_app/measure_generalization.py [N] [seed]
"""

from __future__ import annotations

import random
import sys

from ncpu.synthesis_api.server import SynthConfig, handle_synthesize_request


def make_piecewise(rng: random.Random):
    """A random continuous piecewise-affine f: int -> int (tiered-pricing shape).

    Returns (fn, descriptor). Continuous because each tier continues from where
    the previous left off — exactly how real tier schedules behave."""
    n_tiers = rng.randint(1, 4)
    # increasing breakpoints
    bps = sorted(rng.sample(range(50, 20000), n_tiers - 1)) if n_tiers > 1 else []
    slopes = [rng.choice([0, 1, 2, 3, 5, 10]) for _ in range(n_tiers)]
    base = 0  # value at x=0

    def fn(x: int) -> int:
        # tier index
        edges = [0] + bps
        total = base
        for i in range(n_tiers):
            lo = edges[i]
            hi = bps[i] if i < len(bps) else None
            if hi is not None and x > hi:
                total += slopes[i] * (hi - lo)
            elif x > lo:
                total += slopes[i] * (x - lo)
                break
            else:
                break
        return total

    desc = f"{n_tiers}tier bps={bps} slopes={slopes}"
    return fn, desc, (bps[-1] if bps else 100)


def sample_training(fn, rng: random.Random, max_bp: int):
    """A modest, uncurated training set spanning the range: 0, a few near each
    breakpoint region, and random points. ~12-16 points."""
    hi = max(max_bp * 2, 200)
    xs = {0, hi}
    # spread points across the range
    for _ in range(10):
        xs.add(rng.randint(0, hi))
    # a few small ones so tier 0 is represented
    for _ in range(3):
        xs.add(rng.randint(0, max(1, max_bp)))
    xs = sorted(xs)
    return [(x, fn(x)) for x in xs], hi


def classify(fn, code_py: str, entry: str, hi: int):
    """SOLVED / OVERFIT / FAILED against dense unseen points up to 1.5*hi."""
    if not code_py:
        return "FAILED", None
    ns: dict = {}
    try:
        exec(code_py, ns)  # noqa: S102 — verification only, our own transpile
        g = ns[entry]
    except Exception:
        return "FAILED", None
    bad = 0
    checked = 0
    for x in range(0, int(hi * 1.5) + 1, 7):
        checked += 1
        try:
            if g(x) != fn(x):
                bad += 1
        except Exception:
            bad += 1
    return ("SOLVED" if bad == 0 else "OVERFIT"), bad


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    cfg = SynthConfig(timeout_s=20.0, max_timeout_s=20.0)
    rng = random.Random(seed)

    counts = {"SOLVED": 0, "OVERFIT": 0, "FAILED": 0}
    by_tier: dict[int, dict[str, int]] = {}
    overfit_examples = []

    for i in range(n):
        fn, desc, max_bp = make_piecewise(rng)
        n_tiers = int(desc.split("tier")[0])
        train, hi = sample_training(fn, rng, max_bp)
        entry = f"rule_{i}"
        req = {
            "name": entry,
            "signature": f"fn {entry}(x: i64) -> i64",
            "examples": [{"inputs": [x], "expected": y} for x, y in train],
        }
        sc, pl = handle_synthesize_request(req, cfg)
        code_py = (pl.get("transpiled") or {}).get("python", "") if sc == 200 else ""
        verdict, bad = classify(fn, code_py, entry, hi)
        counts[verdict] += 1
        by_tier.setdefault(n_tiers, {"SOLVED": 0, "OVERFIT": 0, "FAILED": 0})
        by_tier[n_tiers][verdict] += 1
        if verdict == "OVERFIT" and len(overfit_examples) < 6:
            overfit_examples.append((desc, pl.get("method"), bad))
        print(f"  [{i:2}] {verdict:7} ({desc})  method={pl.get('method')}")

    print("\n" + "=" * 60)
    tot = sum(counts.values())
    print(f"RAW nsynth on {tot} unseen piecewise rules (examples only):")
    print(f"  SOLVED  (correct on unseen): {counts['SOLVED']:3}  ({100*counts['SOLVED']/tot:.0f}%)")
    print(f"  OVERFIT (wrong on unseen):   {counts['OVERFIT']:3}  ({100*counts['OVERFIT']/tot:.0f}%)")
    print(f"  FAILED  (no program):        {counts['FAILED']:3}  ({100*counts['FAILED']/tot:.0f}%)")
    print("\n  by #tiers:")
    for t in sorted(by_tier):
        c = by_tier[t]
        tt = sum(c.values())
        print(f"    {t}-tier: solved {c['SOLVED']}/{tt}  overfit {c['OVERFIT']}  failed {c['FAILED']}")
    if overfit_examples:
        print("\n  sample overfits (synthesized, wrong on unseen):")
        for desc, method, bad in overfit_examples:
            print(f"    {desc}  via {method}  ({bad} unseen mismatches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
