#!/usr/bin/env python3
"""
Observe a reference function on random inputs, emit JSONL in the schema
jsonl_harvest expects. This is the first concrete "learn from watching
execution" pipeline — the Rust synthesiser downstream has no access to the
reference function's source, only the I/O pairs this script produces.

The "reference function" stand-in is a Python callable. In the long-running
design this should be replaced with:
  - A function exported by a compiled binary (via the nCPU ARM64 emulator)
  - A fuzzer capture
  - Instrumented production code

What matters is that the I/O shape is what the system sees — not the
implementation. That makes this pipeline programming-language-agnostic.

Usage:
    python3 tools/extract_io.py \
        --function fibonacci \
        --out /tmp/corpus.jsonl \
        [--samples 20] \
        [--range -50:50] \
        [--seed 42]

Then pipe into the harvester:
    ./target/release/jsonl_harvest \
        --in /tmp/corpus.jsonl \
        --out /tmp/learned.jsonl \
        --verbose \
        --max-steps 200

Supported reference functions (the "binaries" we're observing). Kept
deliberately simple so a human can sanity-check the learned Mog program
against the intended behaviour. Extending the set just means adding entries
to REFS below.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Callable, Dict, Tuple


# ─── Reference functions ─────────────────────────────────────────────────────
# Each entry is (signature_source, callable). The signature string is what the
# Rust synthesiser sees as the target type — keep it accurate.

REFS: Dict[str, Tuple[str, Callable[..., int], int]] = {
    # name: (rust_signature, fn, arity)
    "fibonacci": (
        "fn fibonacci(n: i64) -> i64",
        lambda n: _fibonacci(n),
        1,
    ),
    "double_plus_one": (
        "fn double_plus_one(a: i64) -> i64",
        lambda a: 2 * a + 1,
        1,
    ),
    "clamp_0_10": (
        "fn clamp_0_10(a: i64) -> i64",
        lambda a: max(0, min(10, a)),
        1,
    ),
    "triangular": (
        "fn triangular(n: i64) -> i64",
        lambda n: _triangular(n),
        1,
    ),
    "abs_diff": (
        "fn abs_diff(a: i64, b: i64) -> i64",
        lambda a, b: abs(a - b),
        2,
    ),
    "scaled_sum": (
        "fn scaled_sum(a: i64, b: i64) -> i64",
        lambda a, b: 2 * a + 3 * b,
        2,
    ),
}


def _fibonacci(n: int) -> int:
    if n < 0:
        return 0
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _triangular(n: int) -> int:
    if n <= 0:
        return 0
    return n * (n + 1) // 2


# ─── I/O generation ──────────────────────────────────────────────────────────

def parse_range(raw: str) -> Tuple[int, int]:
    """Parse ``"min:max"`` → ``(min, max)`` inclusive. Rejects reversed ranges
    with a clear error so a bad CLI doesn't silently sample nothing."""
    try:
        lo_s, hi_s = raw.split(":")
        lo, hi = int(lo_s), int(hi_s)
    except ValueError as exc:
        raise SystemExit(f"[extract_io] bad --range {raw!r}: expected min:max integers") from exc
    if lo > hi:
        raise SystemExit(f"[extract_io] bad --range {raw!r}: min must be <= max")
    return lo, hi


def sample_inputs(arity: int, lo: int, hi: int, n: int, rng: random.Random) -> list:
    """Draw `n` unique input tuples from [lo, hi]^arity. Falls back to
    duplicates only when the cartesian space is smaller than `n` — which is
    deliberate: the synthesiser tolerates repeats, but we prefer distinct
    examples because they pin down more of the function's shape."""
    space_size = (hi - lo + 1) ** arity
    seen = set()
    out = []
    attempts = 0
    cap = max(n * 20, 200)
    while len(out) < n and attempts < cap:
        attempts += 1
        t = tuple(rng.randint(lo, hi) for _ in range(arity))
        if t in seen:
            continue
        seen.add(t)
        out.append(list(t))
    if len(out) < n:
        # Space was too small; fill with replacement so the caller still gets
        # `n` entries and downstream logic doesn't have to special-case short
        # outputs.
        while len(out) < n:
            out.append([rng.randint(lo, hi) for _ in range(arity)])
    return out


def build_problem_record(name: str, signature: str, examples: list) -> dict:
    return {
        "name": f"{name}_extracted",
        "signature": signature,
        "examples": [{"inputs": ins, "expected": exp} for ins, exp in examples],
    }


# ─── Entry ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--function",
        choices=sorted(REFS),
        required=True,
        help="Reference function to observe.",
    )
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of I/O pairs to capture (default 20).",
    )
    ap.add_argument(
        "--range",
        default="-20:20",
        help="Input sampling range, inclusive (default -20:20).",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42).")
    args = ap.parse_args()

    signature, fn, arity = REFS[args.function]
    lo, hi = parse_range(args.range)
    rng = random.Random(args.seed)

    inputs = sample_inputs(arity, lo, hi, args.samples, rng)
    examples = []
    for ins in inputs:
        try:
            out = fn(*ins)
        except Exception as exc:
            print(
                f"[extract_io] {args.function}{tuple(ins)} raised {exc!r} — skipping",
                file=sys.stderr,
            )
            continue
        if not isinstance(out, int):
            # The Rust synthesiser only accepts i64 outputs. Coerce bools →
            # {0, 1}, refuse anything else with a clear error.
            if isinstance(out, bool):
                out = int(out)
            else:
                raise SystemExit(
                    f"[extract_io] {args.function} returned non-integer {out!r}: "
                    "only i64-returning functions are supported by jsonl_harvest."
                )
        examples.append((ins, out))

    if not examples:
        raise SystemExit("[extract_io] no valid examples produced — refusing to emit empty record")

    record = build_problem_record(args.function, signature, examples)
    with open(args.out, "w") as f:
        f.write(json.dumps(record))
        f.write("\n")

    print(
        f"[extract_io] {args.function}: captured {len(examples)} examples over "
        f"range {lo}..{hi} → {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
