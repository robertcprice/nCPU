"""CLI: complex English in, verified program out.

    python -m ncpu.requirements.cli "Compute the absolute difference of two integers."
    python -m ncpu.requirements.cli --deterministic "add(1,2) -> 3  add(5,7) -> 12"
    python -m ncpu.requirements.cli --json "..."   # machine-readable result

Default proposer is the LLM (needs ANTHROPIC_API_KEY); --deterministic uses
the no-LLM parser (explicit examples only).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from ncpu.requirements.pipeline import resolve
from ncpu.requirements.proposer import DeterministicProposer, LLMProposer


def _render(result) -> str:
    lines = []
    lines.append(f"status: {result.status}   confidence: {result.confidence}")
    if result.ir:
        lines.append(f"entry_point: {result.ir.entry_point}")
        lines.append(f"signature:   {result.ir.signature_str()}")
        if result.ir.invariants:
            lines.append("invariants:  " + "; ".join(result.ir.invariants))
    if result.method:
        lines.append(f"method:      {result.method}")
    if result.holdout_count:
        lines.append(
            f"holdout:     {result.holdout_passed}/{result.holdout_count} held-out "
            f"examples reproduced"
        )
    if result.synth_vs_reference_agree is not None:
        lines.append(
            f"vs reference: {'agree' if result.synth_vs_reference_agree else 'DISAGREE'}"
        )
    if result.transpiled.get("python"):
        lines.append("\n--- synthesized (python) ---")
        lines.append(result.transpiled["python"])
    elif result.program:
        lines.append("\n--- synthesized (mog) ---")
        lines.append(result.program)
    if result.notes:
        lines.append("\nnotes:")
        for n in result.notes:
            if n:
                lines.append(f"  - {n}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("english", help="the natural-language request")
    p.add_argument("--deterministic", action="store_true", help="no-LLM parser")
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--no-cross-check", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    proposer = (
        DeterministicProposer() if args.deterministic else LLMProposer(model=args.model)
    )
    result = resolve(
        args.english,
        proposer=proposer,
        synth_timeout_s=args.timeout,
        cross_check=not args.no_cross_check,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(_render(result))
    return 0 if result.status == "synthesized" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
