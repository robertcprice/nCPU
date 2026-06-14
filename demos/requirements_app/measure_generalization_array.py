"""Honest capability probe for ARRAY-FEATURE composition.

The array-composition solver proposes array REDUCTIONS — len, sum, min, max,
first, last, range, count_positive/zero/even, sum_of_abs, sum_of_squares — and
recovers an exact affine combination of their outputs (plus any trailing scalar
args). Most real array rules are exactly this shape ("total = base + 2*len +
sum"). A 12-reduction basis can fit noise, so this probe is the contract: random
rules of that shape, examples ONLY (no reference, no CEGIS, no curation), SOLVED
only when the program is exactly correct on a batch of UNSEEN arrays. A fit that
matches the training arrays but is wrong on new ones is an OVERFIT.

Usage:  PYTHONPATH=. python3 demos/requirements_app/measure_generalization_array.py [N] [seed]
"""

from __future__ import annotations

import random
import sys

from ncpu.synthesis_api.server import SynthConfig, handle_synthesize_request

REDUCTIONS = {
    "len": lambda a: len(a),
    "sum": lambda a: sum(a),
    "min": lambda a: min(a),
    "max": lambda a: max(a),
    "first": lambda a: a[0],
    "last": lambda a: a[-1],
    "range": lambda a: max(a) - min(a),
    "count_pos": lambda a: sum(1 for x in a if x > 0),
    "count_zero": lambda a: sum(1 for x in a if x == 0),
    "count_even": lambda a: sum(1 for x in a if x % 2 == 0),
    "sum_abs": lambda a: sum(abs(x) for x in a),
    "sum_sq": lambda a: sum(x * x for x in a),
}
KEYS = list(REDUCTIONS)


def make_rule(rng: random.Random):
    """A random affine-over-reductions rule with 0–2 trailing scalar args.
    1–3 reduction terms so it is genuinely sparse (the shape the solver targets);
    all reductions used are defined on non-empty arrays."""
    n_scalars = rng.randint(0, 2)
    c0 = rng.choice([0, 1, 3, 5, -2, 10])
    n_terms = rng.randint(1, 3)
    terms = rng.sample(KEYS, n_terms)
    coeffs = {k: rng.choice([1, 2, 3, 5, -1, -2]) for k in terms}
    scoeffs = [rng.choice([0, 1, 2, -1, 3]) for _ in range(n_scalars)]

    def fn(arr, scalars):
        v = c0 + sum(coeffs[k] * REDUCTIONS[k](arr) for k in terms)
        v += sum(scoeffs[j] * scalars[j] for j in range(n_scalars))
        return v

    desc = f"c0={c0} " + " + ".join(f"{coeffs[k]}*{k}" for k in terms)
    if n_scalars:
        desc += " | scalars=" + str(scoeffs)
    return fn, desc, n_scalars


def rand_array(rng: random.Random):
    n = rng.randint(1, 8)
    return [rng.randint(-9, 12) for _ in range(n)]


def sample_training(fn, n_scalars: int, rng: random.Random):
    """Uncurated training: ~16+ (array, scalars) pairs of varied length/content,
    enough to over-determine a 3-reduction + 2-scalar fit."""
    rows = []
    seen = set()
    n_rows = 18 + 3 * n_scalars
    while len(rows) < n_rows:
        arr = rand_array(rng)
        scalars = [rng.randint(0, 9) for _ in range(n_scalars)]
        key = (tuple(arr), tuple(scalars))
        if key in seen:
            continue
        seen.add(key)
        rows.append((arr, scalars, fn(arr, scalars)))
    return rows


def classify(fn, code_py: str, entry: str, n_scalars: int, rng: random.Random):
    if not code_py:
        return "FAILED", None
    ns: dict = {}
    try:
        exec(code_py, ns)  # noqa: S102
        g = ns[entry]
    except Exception:
        return "FAILED", None
    bad = 0
    for _ in range(60):  # held-out arrays, in-distribution
        arr = rand_array(rng)
        scalars = [rng.randint(0, 9) for _ in range(n_scalars)]
        try:
            got = g(arr, *scalars) if n_scalars else g(arr)
            if got != fn(arr, scalars):
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
        fn, desc, n_scalars = make_rule(rng)
        train = sample_training(fn, n_scalars, rng)
        entry = f"rule_{i}"
        sig_args = "arr: [i64]" + "".join(f", s{j}: i64" for j in range(n_scalars))
        examples = [
            {"inputs": [arr] + scalars, "expected": y} for arr, scalars, y in train
        ]
        req = {"name": entry, "signature": f"fn {entry}({sig_args}) -> i64", "examples": examples}
        sc, pl = handle_synthesize_request(req, cfg)
        code_py = (pl.get("transpiled") or {}).get("python", "") if sc == 200 else ""
        verdict, bad = classify(fn, code_py, entry, n_scalars, random.Random(seed * 100 + i))
        counts[verdict] += 1
        if verdict != "SOLVED" and len(misses) < 10:
            misses.append((verdict, desc, pl.get("method"), bad))
        print(f"  [{i:2}] {verdict:7} ({desc})  method={pl.get('method')}")

    print("\n" + "=" * 60)
    tot = sum(counts.values())
    print(f"RAW nsynth on {tot} unseen array-feature rules (examples only):")
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
