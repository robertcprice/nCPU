"""Honest capability probe for the SEPARABLE-POLYNOMIAL solver.

The affine solvers recover only straight-line rules. Real formulas curve:
area = side², kinetic energy = ½mv², a projectile's height is quadratic in
time. This probe asks the raw synthesizer (examples ONLY — no reference, no
CEGIS, no curation) to recover random separable polynomials

    f(x_1..x_k) = c0 + Σ c_j x_j + Σ d_j x_j²        (1 <= k <= 3)

and scores SOLVED only when the returned program is exactly correct on a DENSE
grid of UNSEEN points. A program that fits the training samples but is wrong
between them is an OVERFIT — the failure mode that matters. Nothing here is
hand-picked to pass.

Usage:  PYTHONPATH=. python3 demos/requirements_app/measure_generalization_polynomial.py [N] [seed]
"""

from __future__ import annotations

import itertools
import random
import sys

from ncpu.synthesis_api.server import SynthConfig, handle_synthesize_request


def make_polynomial(rng: random.Random):
    """A random separable polynomial f: int^k -> int with curvature.

    At least one quadratic coefficient is non-zero so the rule genuinely curves
    (a pure-linear draw would be solvable by the affine path and wouldn't test
    the polynomial solver). Coefficients are small integers — the shape of real
    formulas, not adversarial magnitudes."""
    k = rng.randint(1, 3)
    c0 = rng.choice([0, 1, 2, 3, 5, 10, -1, -3])
    lin = [rng.choice([0, 1, 2, 3, -1, -2]) for _ in range(k)]
    quad = [rng.choice([0, 1, 2, -1]) for _ in range(k)]
    if all(d == 0 for d in quad):  # force curvature
        quad[rng.randrange(k)] = rng.choice([1, 2, -1])

    def fn(*xs: int) -> int:
        return c0 + sum(lin[j] * xs[j] for j in range(k)) + sum(quad[j] * xs[j] * xs[j] for j in range(k))

    desc = f"{k}arg c0={c0} lin={lin} quad={quad}"
    return fn, desc, k


def sample_training(fn, k: int, rng: random.Random):
    """A modest, uncurated training set. Small-magnitude grid plus random
    spread — enough rows to determine 1 + 2k coefficients, not curated to the
    answer. ~12-18 points."""
    base = list(range(0, 6))  # 0..5 so quadratic curvature is visible
    pts = set()
    # a small structured grid (cartesian over a few small values per axis)
    grid_vals = [0, 1, 2, 3, 5]
    for combo in itertools.product(grid_vals, repeat=k):
        pts.add(combo)
        if len(pts) >= 10:
            break
    # random spread to a larger range
    while len(pts) < 12 + 2 * k:
        pts.add(tuple(rng.randint(0, 40) for _ in range(k)))
    rows = [(list(p), fn(*p)) for p in sorted(pts)]
    return rows


def classify(fn, code_py: str, entry: str, k: int):
    """SOLVED / OVERFIT / FAILED on a dense UNSEEN grid (larger than training)."""
    if not code_py:
        return "FAILED", None
    ns: dict = {}
    try:
        exec(code_py, ns)  # noqa: S102 — verification only, our own transpile
        g = ns[entry]
    except Exception:
        return "FAILED", None
    bad = 0
    # unseen points up to 60 (training maxed at 40) on a coarse grid
    axis = list(range(0, 61, 9))
    for combo in itertools.product(axis, repeat=k):
        try:
            if g(*combo) != fn(*combo):
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
    by_arg: dict[int, dict[str, int]] = {}
    overfit_examples = []

    for i in range(n):
        fn, desc, k = make_polynomial(rng)
        train = sample_training(fn, k, rng)
        entry = f"rule_{i}"
        args = ", ".join(f"x{j}: i64" for j in range(k))
        req = {
            "name": entry,
            "signature": f"fn {entry}({args}) -> i64",
            "examples": [{"inputs": xs, "expected": y} for xs, y in train],
        }
        sc, pl = handle_synthesize_request(req, cfg)
        code_py = (pl.get("transpiled") or {}).get("python", "") if sc == 200 else ""
        verdict, bad = classify(fn, code_py, entry, k)
        counts[verdict] += 1
        by_arg.setdefault(k, {"SOLVED": 0, "OVERFIT": 0, "FAILED": 0})
        by_arg[k][verdict] += 1
        if verdict == "OVERFIT" and len(overfit_examples) < 6:
            overfit_examples.append((desc, pl.get("method"), bad))
        print(f"  [{i:2}] {verdict:7} ({desc})  method={pl.get('method')}")

    print("\n" + "=" * 60)
    tot = sum(counts.values())
    print(f"RAW nsynth on {tot} unseen separable polynomials (examples only):")
    print(f"  SOLVED  (correct on unseen): {counts['SOLVED']:3}  ({100*counts['SOLVED']/tot:.0f}%)")
    print(f"  OVERFIT (wrong on unseen):   {counts['OVERFIT']:3}  ({100*counts['OVERFIT']/tot:.0f}%)")
    print(f"  FAILED  (no program):        {counts['FAILED']:3}  ({100*counts['FAILED']/tot:.0f}%)")
    print("\n  by #args:")
    for kk in sorted(by_arg):
        c = by_arg[kk]
        tt = sum(c.values())
        print(f"    {kk}-arg: solved {c['SOLVED']}/{tt}  overfit {c['OVERFIT']}  failed {c['FAILED']}")
    if overfit_examples:
        print("\n  sample overfits (synthesized, wrong on unseen):")
        for desc, method, bad in overfit_examples:
            print(f"    {desc}  via {method}  ({bad} unseen mismatches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
