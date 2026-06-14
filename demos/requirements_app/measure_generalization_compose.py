"""Honest capability probe for COMPOSITIONAL ("think-in-code") synthesis.

The composition solver proposes derived features — squares, cross-terms a·b,
modulo x%m, floor-div x/m — and recovers an exact affine rule over them. A rich
feature basis is exactly the kind of thing that can fit noise, so this probe is
the contract: it generates random rules of that shape, gives the synthesizer the
examples ONLY (no reference, no CEGIS, no curation), and scores SOLVED only when
the program is exactly correct on a DENSE grid of UNSEEN, in-distribution points.
A fit that matches the training samples but is wrong between them is an OVERFIT —
the failure mode that matters most here. Nothing is hand-picked to pass.

Usage:  PYTHONPATH=. python3 demos/requirements_app/measure_generalization_compose.py [N] [seed]
"""

from __future__ import annotations

import itertools
import random
import sys

from ncpu.synthesis_api.server import SynthConfig, handle_synthesize_request


def make_composed(rng: random.Random):
    """A random rule built from derived features the solver can express:
    f(x) = c0 + Σ c_k·g_k(x), each g_k one of {x_j, x_j², x_i·x_j, x_j%m, x_j/m}.
    1–3 features so it is genuinely sparse (the shape the solver targets)."""
    k = rng.randint(1, 3)  # arity
    c0 = rng.choice([0, 1, 3, 5, -2, 10])
    n_feats = rng.randint(1, 3)
    feats = []  # (descriptor, fn-over-args)
    kinds = ["raw", "sq", "mod", "div"] + (["cross"] if k >= 2 else [])
    used = set()
    while len(feats) < n_feats:
        kind = rng.choice(kinds)
        c = rng.choice([1, 2, 3, 5, -1, -2])
        if kind == "raw":
            j = rng.randrange(k)
            key = ("raw", j)
            g = (lambda j: lambda xs: xs[j])(j)
            d = f"{c}*x{j}"
        elif kind == "sq":
            j = rng.randrange(k)
            key = ("sq", j)
            g = (lambda j: lambda xs: xs[j] * xs[j])(j)
            d = f"{c}*x{j}^2"
        elif kind == "cross":
            i, j = sorted(rng.sample(range(k), 2))
            key = ("cross", i, j)
            g = (lambda i, j: lambda xs: xs[i] * xs[j])(i, j)
            d = f"{c}*x{i}*x{j}"
        elif kind == "mod":
            j = rng.randrange(k)
            m = rng.choice([2, 3, 4, 5, 7])
            key = ("mod", j, m)
            g = (lambda j, m: lambda xs: xs[j] % m)(j, m)
            d = f"{c}*(x{j}%{m})"
        else:  # div
            j = rng.randrange(k)
            m = rng.choice([2, 3, 4, 5, 10])
            key = ("div", j, m)
            g = (lambda j, m: lambda xs: xs[j] // m)(j, m)
            d = f"{c}*(x{j}/{m})"
        if key in used:
            continue
        used.add(key)
        feats.append((c, g, d))

    def fn(*xs: int) -> int:
        return c0 + sum(c * g(xs) for c, g, _ in feats)

    desc = f"{k}arg c0={c0} " + " + ".join(d for _, _, d in feats)
    return fn, desc, k


def sample_training(fn, k: int, rng: random.Random):
    """Uncurated training over the 0..30 box: a small grid plus random spread,
    enough rows that even a 3-feature fit is over-determined (the solver needs
    features+3 examples)."""
    pts = set()
    grid_vals = [0, 1, 2, 3, 5, 7, 10, 15]
    for combo in itertools.product(grid_vals, repeat=k):
        pts.add(combo)
        if len(pts) >= 16:
            break
    while len(pts) < 20 + 3 * k:
        pts.add(tuple(rng.randint(0, 30) for _ in range(k)))
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
    axis = [4, 9, 13, 18, 22, 27]  # held-out, in-distribution (0..30 box)
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
    misses = []

    for i in range(n):
        fn, desc, k = make_composed(rng)
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
        if verdict != "SOLVED" and len(misses) < 10:
            misses.append((verdict, desc, pl.get("method"), bad))
        print(f"  [{i:2}] {verdict:7} ({desc})  method={pl.get('method')}")

    print("\n" + "=" * 60)
    tot = sum(counts.values())
    print(f"RAW nsynth on {tot} unseen composed rules (examples only):")
    print(f"  SOLVED  (correct on unseen): {counts['SOLVED']:3}  ({100*counts['SOLVED']/tot:.0f}%)")
    print(f"  OVERFIT (wrong on unseen):   {counts['OVERFIT']:3}  ({100*counts['OVERFIT']/tot:.0f}%)")
    print(f"  FAILED  (no program):        {counts['FAILED']:3}  ({100*counts['FAILED']/tot:.0f}%)")
    print("\n  by #args:")
    for kk in sorted(by_arg):
        c = by_arg[kk]
        tt = sum(c.values())
        print(f"    {kk}-arg: solved {c['SOLVED']}/{tt}  overfit {c['OVERFIT']}  failed {c['FAILED']}")
    if misses:
        print("\n  sample misses:")
        for verdict, desc, method, bad in misses:
            print(f"    {verdict}: {desc}  via {method}  ({bad} mismatches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
