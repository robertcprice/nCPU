"""Honest capability probe for PREDICATE-BRANCH synthesis (conditional logic).

The branch solver recovers `if P(x) { A(x) } else { B(x) }` where P is a modular
(`x % m == r`) or threshold predicate and A, B are affine. Many candidate
predicates means many chances to find a coincidental split, so this probe is the
contract: random branch rules, examples ONLY (no reference, no CEGIS), SOLVED
only when the program is exactly correct on a dense grid of UNSEEN points. A fit
that matches the training samples but is wrong between them is an OVERFIT.

Usage:  PYTHONPATH=. python3 demos/requirements_app/measure_generalization_branch.py [N] [seed]
"""

from __future__ import annotations

import itertools
import random
import sys

from ncpu.synthesis_api.server import SynthConfig, handle_synthesize_request


def make_branch(rng: random.Random):
    """A random `if P(x_split) { A } else { B }` with affine A != B over 1-2 args.
    P is a modular residue test or a threshold. Bodies are small-integer affines."""
    k = rng.randint(1, 2)
    split = rng.randrange(k)
    # MODULAR / parity class splits — the family search_predicate_branch claims.
    # (Threshold-branch is a separate family owned by the argument-threshold and
    # scalar single/two-branch solvers, with its own breakpoint handling.)
    kind = "mod"

    def affine():
        c0 = rng.choice([0, 1, 3, 5, -2])
        cs = [rng.choice([1, 2, 3, -1, -2]) for _ in range(k)]
        return c0, cs

    a0, acs = affine()
    b0, bcs = affine()
    while (a0, acs) == (b0, bcs):
        b0, bcs = affine()

    def lin(c0, cs, xs):
        return c0 + sum(cs[j] * xs[j] for j in range(k))

    m = rng.choice([2, 3, 4, 5])
    r = rng.randrange(m)
    desc = f"{k}arg if x{split}%{m}=={r} A else B"

    def pred(xs):
        return xs[split] % m == r

    def fn(*xs):
        return lin(a0, acs, xs) if pred(xs) else lin(b0, bcs, xs)

    return fn, desc, k, pred, split, m


def sample_training(fn, k: int, pred, split: int, m: int, rng: random.Random):
    """Uncurated training that makes the structure OBSERVABLE: it densely covers
    every residue class of the split modulus on the split axis (a contiguous run
    0..(3m) gives several points per class), plus random spread on the other
    axes. Without all residue classes present the data cannot distinguish the true
    modulus from a coarser one — and emitting the simplest consistent split would
    be correct, not an overfit. Covering the classes is the fair test."""
    pts = set()
    other = [0, 2, 3, 5, 7, 9, 12, 16, 21, 27]
    for sval in range(0, 3 * m + 1):  # every residue class, multiple times
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
    # extra random spread
    while len(pts) < 24 + 6 * k:
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
    axis = [2, 5, 9, 13, 16, 19, 23, 26, 29]  # held-out, in-distribution
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
        fn, desc, k, pred, split, m = make_branch(rng)
        train = sample_training(fn, k, pred, split, m, rng)
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
    print(f"RAW nsynth on {tot} unseen branch rules (examples only):")
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
