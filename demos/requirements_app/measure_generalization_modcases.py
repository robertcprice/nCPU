"""Honest capability probe for MODULAR CASE ANALYSIS (search_modular_cases).

Recovers `match x_j % m { 0 => A_0, 1 => A_1, ... }` — a distinct affine per
residue class. Random m-way case rules, examples ONLY, SOLVED only when exactly
correct on UNSEEN points. Training covers every residue class densely (a class
never sampled is unrecoverable by anyone), so this is the fair test.

Usage:  PYTHONPATH=. python3 demos/requirements_app/measure_generalization_modcases.py [N] [seed]
"""

from __future__ import annotations

import itertools
import random
import sys

from ncpu.synthesis_api.server import SynthConfig, handle_synthesize_request


def make_modcase(rng: random.Random):
    k = rng.randint(1, 2)
    split = rng.randrange(k)
    m = rng.choice([3, 4, 5, 6, 7])

    def affine():
        return rng.choice([0, 1, 3, 5, -2]), [rng.choice([1, 2, 3, -1, -2]) for _ in range(k)]

    pieces = [affine() for _ in range(m)]
    # ensure not all identical
    while all(p == pieces[0] for p in pieces):
        pieces = [affine() for _ in range(m)]

    def fn(*xs):
        c0, cs = pieces[xs[split] % m]
        return c0 + sum(cs[j] * xs[j] for j in range(k))

    desc = f"{k}arg match x{split}%{m} ({m} affines)"
    return fn, desc, k, split, m


def sample_training(fn, k: int, split: int, m: int, rng: random.Random):
    """Cover every residue class densely on the split axis (a contiguous run so
    each class has several points) + random spread on the other axes."""
    pts = set()
    other = [0, 2, 3, 5, 7, 9, 12, 16, 21, 27]
    reps = max(4, k + 3)
    for sval in range(0, m * reps + 1):
        if k == 1:
            pts.add((sval,))
        else:
            for _ in range(2):
                combo = [0] * k
                combo[split] = sval
                for j in range(k):
                    if j != split:
                        combo[j] = rng.choice(other)
                pts.add(tuple(combo))
    while len(pts) < m * (k + 2) + 10:
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
    axis = [1, 4, 8, 11, 15, 19, 22, 26, 31, 37]  # held-out, in-distribution
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
    misses = []

    for i in range(n):
        fn, desc, k, split, m = make_modcase(rng)
        train = sample_training(fn, k, split, m, rng)
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
        if verdict != "SOLVED" and len(misses) < 10:
            misses.append((verdict, desc, pl.get("method"), bad))
        print(f"  [{i:2}] {verdict:7} ({desc})  method={pl.get('method')}")

    print("\n" + "=" * 60)
    tot = sum(counts.values())
    print(f"RAW nsynth on {tot} unseen modular-case rules (examples only):")
    print(f"  SOLVED  (correct on unseen): {counts['SOLVED']:3}  ({100*counts['SOLVED']/tot:.0f}%)")
    print(f"  OVERFIT (wrong on unseen):   {counts['OVERFIT']:3}  ({100*counts['OVERFIT']/tot:.0f}%)")
    print(f"  FAILED  (no program):        {counts['FAILED']:3}  ({100*counts['FAILED']/tot:.0f}%)")
    if misses:
        print("\n  sample misses:")
        for verdict, desc, method, bad in misses:
            print(f"    {verdict}: {desc}  via {method}  ({bad} mismatches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
