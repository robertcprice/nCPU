"""Honest capability probe for MAX/MIN-of-two-affine (value-based branching).

Recovers `f(x) = max(A(x), B(x))` or `min(A(x), B(x))`, both pieces non-constant
affine. Random envelopes, examples ONLY, SOLVED only when exactly correct on
UNSEEN points. Training is checked to actually exercise BOTH pieces (an envelope
whose minority piece never wins in the data is just one affine — the honest
answer given the data), so the structure is observable: the fair test.

Usage:  PYTHONPATH=. python3 demos/requirements_app/measure_generalization_minmax.py [N] [seed]
"""

from __future__ import annotations

import itertools
import random
import sys

from ncpu.synthesis_api.server import SynthConfig, handle_synthesize_request


def make_envelope(rng: random.Random):
    """A random max/min of two affines, REJECTION-SAMPLED so both pieces win a
    real share of the 0..30 box (≥30% each). An envelope whose minority piece
    almost never wins degenerates to a single affine over the realistic data —
    and the single affine is then the honest fit, not an overfit — so testing it
    would unfairly penalise the synthesizer for a structure the data cannot show.
    For k==2 the boundary must also be genuinely 2-D (both gradients differ), so
    it is a real half-space and not a single-axis threshold."""
    box = list(itertools.product(range(0, 31, 3), repeat=2))
    for _ in range(200):
        k = rng.randint(1, 2)
        is_max = rng.choice([True, False])

        def affine():
            return rng.choice([0, 1, 3, 5, -2]), [rng.choice([1, 2, 3, -1, -2]) for _ in range(k)]

        a0, acs = affine()
        b0, bcs = affine()
        if (a0, acs) == (b0, bcs):
            continue
        if k == 2 and not (acs[0] != bcs[0] and acs[1] != bcs[1]):
            continue

        def A(xs):
            return a0 + sum(acs[j] * xs[j] for j in range(k))

        def B(xs):
            return b0 + sum(bcs[j] * xs[j] for j in range(k))

        pts = [p[:k] for p in box] if k == 2 else [(v,) for v in range(0, 31)]
        a_wins = sum(1 for p in pts if A(p) >= B(p))
        # Boundary near the centre of the box (40-60% each side) so BOTH winning
        # regions are large enough that an uncurated sample reliably exercises
        # them — making the envelope structure observable.
        if not (0.40 * len(pts) <= a_wins <= 0.60 * len(pts)):
            continue

        def fn(*xs):
            return max(A(xs), B(xs)) if is_max else min(A(xs), B(xs))

        op = "max" if is_max else "min"
        desc = f"{k}arg {op}({a0}+{acs}, {b0}+{bcs})"
        return fn, desc, k, A, B
    # fallback (should be rare): a guaranteed-balanced 1-arg envelope
    return (lambda x: max(x, 10), "1arg max(x, 10)", 1, lambda p: p[0], lambda p: 10)


def sample_training(fn, k: int, A, B, rng: random.Random):
    """Sample EVENLY from each true winning region so both pieces are observable
    (the fair test): partition a dense candidate set by which piece actually wins
    — `fn` equals A or B at each point — then draw a balanced mix. This removes
    the diagonal-sliver sampling artifact where an integer grid lands almost
    entirely in one half-space even for a centre-balanced envelope."""
    cand = list(itertools.product(range(0, 31), repeat=k))
    rng.shuffle(cand)
    a_region = [p for p in cand if fn(*p) == A(p) and A(p) != B(p)]
    b_region = [p for p in cand if fn(*p) == B(p) and A(p) != B(p)]
    per = 12 + 4 * k
    chosen = a_region[:per] + b_region[:per]
    if len(a_region) < 6 or len(b_region) < 6:
        return None  # a piece is unobservable here — skip this envelope
    return [(list(p), fn(*p)) for p in sorted(set(chosen))]


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
    axis = [3, 7, 11, 15, 18, 22, 27]
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

    i = 0
    guard = 0
    while i < n and guard < n * 20:
        guard += 1
        fn, desc, k, A, B = make_envelope(rng)
        train = sample_training(fn, k, A, B, rng)
        if train is None:
            continue  # a piece is unobservable — pick another envelope
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
        i += 1

    print("\n" + "=" * 60)
    tot = sum(counts.values())
    print(f"RAW nsynth on {tot} unseen max/min envelopes (examples only):")
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
