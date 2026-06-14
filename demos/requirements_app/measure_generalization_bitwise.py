"""Honest capability probe for BITWISE-structured synthesis (search_bitwise).

Recovers masks/sets/toggles (a&m, a|m, a^m), pairwise combines (a&b, a|b, a^b),
and affine wrappers (c0 + c1*(a&m)). Random bitwise rules, examples ONLY, SOLVED
only when exactly correct on UNSEEN points. A rich mask basis can fit noise, so
0 overfit is the contract.

Usage:  PYTHONPATH=. python3 demos/requirements_app/measure_generalization_bitwise.py [N] [seed]
"""

from __future__ import annotations

import random
import sys

from ncpu.synthesis_api.server import SynthConfig, handle_synthesize_request

OPS = {"&": lambda a, b: a & b, "|": lambda a, b: a | b, "^": lambda a, b: a ^ b}


def make_bitrule(rng: random.Random):
    kind = rng.choice(["mask", "mask", "pairwise", "affine_mask"])
    if kind == "pairwise":
        k = 2
        op = rng.choice("&|^")
        f = (lambda op: lambda *xs: OPS[op](xs[0], xs[1]))(op)
        desc = f"2arg x0 {op} x1"
    elif kind == "affine_mask":
        k = rng.randint(1, 2)
        j = rng.randrange(k)
        m = rng.choice([1, 3, 7, 15, 6, 12, 5])
        op = rng.choice("&|^")
        c1 = rng.choice([1, 2, 3, -1])
        c0 = rng.choice([0, 1, 5, -2])
        f = (lambda op, j, m, c1, c0: lambda *xs: c0 + c1 * OPS[op](xs[j], m))(op, j, m, c1, c0)
        desc = f"{k}arg {c0}+{c1}*(x{j} {op} {m})"
    else:  # mask
        k = rng.randint(1, 2)
        j = rng.randrange(k)
        m = rng.choice([1, 3, 7, 15, 31, 6, 12, 10, 5])
        op = rng.choice("&|^")
        f = (lambda op, j, m: lambda *xs: OPS[op](xs[j], m))(op, j, m)
        desc = f"{k}arg x{j} {op} {m}"
    return f, desc, k


def sample_training(fn, k: int, rng: random.Random):
    pts = set()
    base = list(range(0, 16))  # cover low bits densely so masks are observable
    for v in base:
        pts.add((v,) if k == 1 else (v, rng.randint(0, 31)))
    while len(pts) < 18 + 6 * k:
        pts.add(tuple(rng.randint(0, 63) for _ in range(k)))
    return [(list(p), fn(*p)) for p in sorted(pts)]


def classify(fn, code_py: str, entry: str, k: int, rng: random.Random):
    if not code_py:
        return "FAILED", None
    ns: dict = {}
    try:
        exec(code_py, ns)  # noqa: S102
        g = ns[entry]
    except Exception:
        return "FAILED", None
    bad = 0
    for _ in range(60):  # unseen, in-distribution
        xs = [rng.randint(0, 63) for _ in range(k)]
        try:
            if g(*xs) != fn(*xs):
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
        fn, desc, k = make_bitrule(rng)
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
        verdict, bad = classify(fn, code_py, entry, k, random.Random(seed * 100 + i))
        counts[verdict] += 1
        if verdict != "SOLVED" and len(misses) < 10:
            misses.append((verdict, desc, pl.get("method"), bad))
        print(f"  [{i:2}] {verdict:7} ({desc})  method={pl.get('method')}")

    print("\n" + "=" * 60)
    tot = sum(counts.values())
    print(f"RAW nsynth on {tot} unseen bitwise rules (examples only):")
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
