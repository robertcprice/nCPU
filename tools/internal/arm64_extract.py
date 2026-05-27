#!/usr/bin/env python3
"""
Architecture-tagged I/O extractor — scaffolding for the cross-ISA study.

This is the shape the ncpu ARM64 emulator will eventually plug into: a
script that runs a compiled function through an emulator, captures
(input, output) pairs, and emits JSONL ready for `jsonl_harvest` or
`multi_corpus_clusters`.

The CURRENT backend is a Python callable (same one `extract_io.py` uses).
The intent is that swapping the backend to a real emulator is a
one-function replace — `execute_on_arch` below. Everything else stays.

The `--architecture` label is pass-through metadata. Once multiple
backends exist (arm64, riscv, x86), running this script N times with
different labels produces the multi-corpus input that
`multi_corpus_clusters` turns into cross-ISA invariant families.

Usage:
    python3 tools/arm64_extract.py \
        --architecture arm64 \
        --function fibonacci \
        --out /tmp/arm64_fib.jsonl \
        [--samples 20] [--range -20:20] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Callable, Dict, List, Tuple

# Reuse the reference-function registry from extract_io.py so the two
# extractors stay in lock-step. Importing it this way avoids duplicating
# the list of "functions the system can observe".
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from extract_io import REFS, parse_range, sample_inputs, build_problem_record  # noqa: E402


def execute_on_arch(
    architecture: str,
    fn: Callable[..., int],
    inputs: List[int],
) -> int:
    """Swap point for a real emulator. Today this just calls the Python
    callable; tomorrow it dispatches to the matching architecture backend.

    The architecture label is passed so future implementations can route:
      if architecture == "arm64": return run_arm64_emulator(fn, inputs)
      if architecture == "riscv": return run_riscv_emulator(fn, inputs)
      ...

    Today the implementation is backend-agnostic — every architecture
    returns the same output (from the Python reference). That's fine for
    wiring up the pipeline; real behavioural divergence shows up when
    actual emulators replace this stub.
    """
    _ = architecture  # preserved in JSONL, not used by the Python stub
    return fn(*inputs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--architecture", required=True, help="Label for this trace source (e.g. 'arm64').")
    ap.add_argument("--function", choices=sorted(REFS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--range", default="-20:20")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    signature, fn, arity = REFS[args.function]
    lo, hi = parse_range(args.range)
    rng = random.Random(args.seed)

    inputs = sample_inputs(arity, lo, hi, args.samples, rng)
    examples: List[Tuple[List[int], int]] = []
    for ins in inputs:
        try:
            out = execute_on_arch(args.architecture, fn, ins)
        except Exception as exc:
            print(
                f"[arm64_extract] {args.architecture}:{args.function}"
                f"{tuple(ins)} raised {exc!r} — skipping",
                file=sys.stderr,
            )
            continue
        if isinstance(out, bool):
            out = int(out)
        if not isinstance(out, int):
            raise SystemExit(
                f"[arm64_extract] {args.function} returned non-int {out!r} — "
                "jsonl_harvest only accepts i64 outputs."
            )
        examples.append((ins, out))

    if not examples:
        raise SystemExit("[arm64_extract] no valid examples; refusing to emit empty record")

    record = build_problem_record(args.function, signature, examples)
    # Tag the record with architecture + seed so downstream tools can
    # group / filter. The `multi_corpus_clusters` binary reads --source
    # LABEL:PATH so this tag is currently just for provenance.
    record["architecture"] = args.architecture
    record["seed"] = args.seed

    with open(args.out, "w") as f:
        f.write(json.dumps(record))
        f.write("\n")

    print(
        f"[arm64_extract] {args.architecture}:{args.function} → "
        f"{len(examples)} examples over range {lo}..{hi} → {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
