"""Honest capability probe for the CLAMPED-AFFINE solver.

Real rules saturate: a minimum fee ("never below X"), a spend cap ("never above
Y"), a two-sided band. The affine solvers refuse the moment the data stops being
one straight line. This probe asks the raw synthesizer (examples ONLY — no
reference, no CEGIS, no curation) to recover random clamped-affine rules

    floor:  max(lo, A(x))
    cap:    min(hi, A(x))
    band:   min(hi, max(lo, A(x)))        with A(x) = c0 + Σ c_j·x_j  (1 <= k <= 3)

and scores SOLVED only when the returned program is exactly correct on a DENSE
grid of UNSEEN points spanning both the saturated region and the active region.
A program that fits the training samples but is wrong between them is an OVERFIT.
Nothing here is hand-picked to pass.

Usage:  PYTHONPATH=. python3 demos/requirements_app/measure_generalization_clamp.py [N] [seed]
"""

from __future__ import annotations

import itertools
import random
import sys

from ncpu.synthesis_api.server import SynthConfig, handle_synthesize_request


def make_clamp(rng: random.Random):
    """A random clamped-affine f: int^k -> int. The bounds are chosen relative to
    the affine's range over the sampled box so the clamp is genuinely active (the
    output actually saturates on some inputs) — otherwise the rule degenerates to
    plain affine and would not test the clamp solver."""
    k = rng.randint(1, 3)
    kind = rng.choice(["floor", "cap", "band"])
    c0 = rng.choice([0, 5, 10, -5, 20])
    coeffs = [rng.choice([1, 2, 3, 5, -1, -2]) for _ in range(k)]
    if all(c == 0 for c in coeffs):
        coeffs[rng.randrange(k)] = rng.choice([1, 2, -1])

    def aff(*xs: int) -> int:
        return c0 + sum(coeffs[j] * xs[j] for j in range(k))

    # affine range over the 0..40 sampling box (corners suffice for a monotone-
    # per-axis affine)
    corners = list(itertools.product([0, 40], repeat=k))
    vals = sorted(aff(*c) for c in corners)
    amin, amax = vals[0], vals[-1]
    span = max(1, amax - amin)
    lo = amin + span // 4
    hi = amax - span // 4
    if lo >= hi:  # too tight to form a band; widen
        lo, hi = amin + span // 5, amax - span // 5

    if kind == "floor":
        fn = lambda *xs: max(lo, aff(*xs))  # noqa: E731
        desc = f"{k}arg floor lo={lo} c0={c0} c={coeffs}"
    elif kind == "cap":
        fn = lambda *xs: min(hi, aff(*xs))  # noqa: E731
        desc = f"{k}arg cap hi={hi} c0={c0} c={coeffs}"
    else:
        fn = lambda *xs: min(hi, max(lo, aff(*xs)))  # noqa: E731
        desc = f"{k}arg band [{lo},{hi}] c0={c0} c={coeffs}"
    return fn, desc, k, kind


def sample_training(fn, k: int, rng: random.Random):
    """Uncurated training set over the 0..40 box. Includes ALL 2^k corners of the
    box so the affine's extremes — and therefore both saturation regions of the
    clamp — are actually observed (a clamp whose bound never appears in the data
    is unrecoverable by anyone; real tiered data shows the cap because customers
    hit it). On top of the corners: a small interior grid plus random spread, so
    the inner affine is over-determined and not curated to the answer."""
    pts = set(itertools.product([0, 40], repeat=k))  # both affine extremes
    grid_vals = [0, 5, 10, 20, 40]
    for combo in itertools.product(grid_vals, repeat=k):
        pts.add(combo)
        if len(pts) >= 14 + 2 ** k:
            break
    while len(pts) < 18 + 2 * k:
        pts.add(tuple(rng.randint(0, 40) for _ in range(k)))
    return [(list(p), fn(*p)) for p in sorted(pts)]


def classify(fn, code_py: str, entry: str, k: int):
    if not code_py:
        return "FAILED", None
    ns: dict = {}
    try:
        exec(code_py, ns)  # noqa: S102
        g = ns[entry]
    except Exception:
        return "FAILED", None
    bad = 0
    # Held-out points WITHIN the 0..40 training box (the training grid uses
    # {0,5,10,20,40} plus randoms; these odd values are almost all unseen). We
    # deliberately do NOT extrapolate past the box: a clamp's bound is only
    # recoverable where the output is observed to saturate, so testing on inputs
    # larger than any training point would penalise the synthesizer for failing
    # to invent a bound the data never shows — not a generalization failure but
    # an unobservable one. In-distribution held-out is the honest test.
    axis = [3, 7, 13, 17, 23, 27, 33, 37]
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
    by_kind: dict[str, dict[str, int]] = {}
    overfit_examples = []

    for i in range(n):
        fn, desc, k, kind = make_clamp(rng)
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
        by_kind.setdefault(kind, {"SOLVED": 0, "OVERFIT": 0, "FAILED": 0})
        by_kind[kind][verdict] += 1
        if verdict != "SOLVED" and len(overfit_examples) < 8:
            overfit_examples.append((verdict, desc, pl.get("method"), bad))
        print(f"  [{i:2}] {verdict:7} ({desc})  method={pl.get('method')}")

    print("\n" + "=" * 60)
    tot = sum(counts.values())
    print(f"RAW nsynth on {tot} unseen clamped-affine rules (examples only):")
    print(f"  SOLVED  (correct on unseen): {counts['SOLVED']:3}  ({100*counts['SOLVED']/tot:.0f}%)")
    print(f"  OVERFIT (wrong on unseen):   {counts['OVERFIT']:3}  ({100*counts['OVERFIT']/tot:.0f}%)")
    print(f"  FAILED  (no program):        {counts['FAILED']:3}  ({100*counts['FAILED']/tot:.0f}%)")
    print("\n  by kind:")
    for kk in sorted(by_kind):
        c = by_kind[kk]
        tt = sum(c.values())
        print(f"    {kk:5}: solved {c['SOLVED']}/{tt}  overfit {c['OVERFIT']}  failed {c['FAILED']}")
    if overfit_examples:
        print("\n  sample misses:")
        for verdict, desc, method, bad in overfit_examples:
            print(f"    {verdict}: {desc}  via {method}  ({bad} unseen mismatches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
