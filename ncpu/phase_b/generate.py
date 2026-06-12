"""Generate the Phase B training corpus from verified sources.

Two real sources, unified into :class:`CorpusRecord`:

1. **egdc_factory** — the differentiable-Mog-compiler benchmark
   (`egdc.mog.benchmark`, 63 factories × variants). Each factory yields a
   problem with an NL ``description``, a Mog ``reference_solution``, and
   concrete ``test_cases``. The reference is *verified* in-process via the
   pure-Python Mog interpreter (no external toolchain) before the record
   is emitted, so ``verified=True`` is earned, not assumed.

2. **autoresearch_humaneval** — the verified HumanEval solves from the
   autoresearch loop (`solved_programs.jsonl` joined with the work-queue's
   parsed ``io_pairs``). Programs are Python; ``verified`` carries the
   cascade's verifier verdict.

CLI::

    python -m ncpu.phase_b.generate --source egdc --variants 5 --out corpus.jsonl
    python -m ncpu.phase_b.generate --source autoresearch \\
        --solved training_results/realworld_vastai/solved_programs.jsonl \\
        --queue .nCPU_autoresearch/humaneval_queue.jsonl --out corpus.jsonl
    python -m ncpu.phase_b.generate --source all --out corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterator, Optional

from ncpu.phase_b.corpus import CorpusRecord, summarize, write_corpus


# ---------------------------------------------------------------------------
# source 1 — egdc.mog benchmark factories (Mog programs, in-process verified)
# ---------------------------------------------------------------------------

def _parse_expected(raw: str) -> Any:
    """Mog stdout is a string; recover an int/list when it is one, else keep
    the raw string (so list/tuple/bool outputs survive as their printed form)."""
    s = raw.strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        val = json.loads(s)
        return val
    except (ValueError, json.JSONDecodeError):
        return s


def egdc_records(
    *, variants: int = 5, seed: int = 42
) -> Iterator[CorpusRecord]:
    """Yield one verified record per (factory, variant).

    Verification: run ``reference_solution + wrapper_template`` through the
    pure-Python Mog interpreter and confirm its printed lines match the
    factory's declared ``test_cases`` exactly. A factory whose reference
    fails to reproduce its own cases is emitted with ``verified=False`` (it
    should never happen for the shipped benchmark, but the corpus stays
    honest if a factory regresses)."""
    from egdc.mog import benchmark as B
    from egdc.mog.lang.interpreter import interpret

    for factory in B.PROBLEM_FACTORIES:
        for variant in range(variants):
            rng = random.Random((hash(factory.__name__) ^ (seed * 1000 + variant)) & 0xFFFFFFFF)
            problem = factory(rng, variant)

            io_pairs = [
                {"inputs": list(args), "expected": _parse_expected(exp)}
                for (args, exp) in problem.test_cases
            ]

            verified = False
            try:
                full = problem.reference_solution + "\n" + problem.wrapper_template
                result = interpret(full)
                if result.success:
                    got = (result.output or "").strip().splitlines()
                    exp = [str(tc[1]).strip() for tc in problem.test_cases]
                    verified = got == exp
            except Exception:  # noqa: BLE001 — interpreter failure → unverified
                verified = False

            yield CorpusRecord(
                source="egdc_factory",
                task_id=problem.name,
                nl_prompt=problem.description.strip(),
                entry_point=problem.name.rsplit("_v", 1)[0],
                signature=problem.signature.strip(),
                program_lang="mog",
                program_src=problem.reference_solution.strip(),
                io_pairs=io_pairs,
                category=getattr(problem, "category", "uncategorized"),
                verified=verified,
            )


# ---------------------------------------------------------------------------
# source 2 — autoresearch verified HumanEval solves (Python programs)
# ---------------------------------------------------------------------------

def autoresearch_records(
    *, solved_path: Path, queue_path: Optional[Path] = None
) -> Iterator[CorpusRecord]:
    """Yield one record per verified solve in ``solved_path``.

    The solve's ``program_python`` is a continuation of the queue prompt's
    ``def`` stub; we join on ``task_id`` to recover the NL prompt and the
    parsed ``io_pairs`` (the solve file alone carries neither)."""
    queue: dict[str, dict[str, Any]] = {}
    if queue_path and queue_path.exists():
        with open(queue_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                queue[item["task_id"]] = item

    if not solved_path.exists():
        return

    with open(solved_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            solved = json.loads(line)
            tid = solved.get("task_id", "")
            q = queue.get(tid, {})
            prompt = q.get("prompt", "")
            # NL prompt = the docstring body if present, else the raw prompt.
            nl = _docstring_of(prompt) or prompt.strip() or solved.get("entry_point", tid)
            io_pairs = [
                {"inputs": p.get("args", []), "expected": _literal(p.get("expected_repr"))}
                for p in (q.get("io_pairs") or [])
            ]
            entry = solved.get("entry_point") or q.get("entry_point") or tid
            # program: prompt stub + verified continuation = a runnable function
            program = (prompt + solved.get("program_python", "")) if prompt else solved.get(
                "program_python", ""
            )

            yield CorpusRecord(
                source="autoresearch_humaneval",
                task_id=tid,
                nl_prompt=nl,
                entry_point=entry,
                signature=_def_line(program, entry),
                program_lang="python",
                program_src=program.strip(),
                io_pairs=io_pairs,
                category=solved.get("source_benchmark", "humaneval"),
                verified=bool(solved.get("verifier_passed", False)),
            )


def _docstring_of(prompt: str) -> str:
    import re

    m = re.search(r'"""(.*?)"""', prompt, re.S)
    if m:
        return " ".join(m.group(1).split())
    return ""


def _def_line(program: str, entry: str) -> str:
    for line in program.splitlines():
        if line.strip().startswith(f"def {entry}"):
            return line.strip().rstrip(":")
    return f"def {entry}(...)"


def _literal(repr_str: Optional[str]) -> Any:
    if repr_str is None:
        return None
    import ast

    try:
        return ast.literal_eval(repr_str)
    except (ValueError, SyntaxError):
        return repr_str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", choices=["egdc", "autoresearch", "all"], default="egdc")
    p.add_argument("--variants", type=int, default=5, help="egdc variants per factory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--solved",
        type=Path,
        default=Path("training_results/realworld_vastai/solved_programs.jsonl"),
    )
    p.add_argument(
        "--queue",
        type=Path,
        default=Path(".nCPU_autoresearch/humaneval_queue.jsonl"),
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--only-verified",
        action="store_true",
        help="drop records whose program did not reproduce its io_pairs",
    )
    args = p.parse_args(argv)

    records: list[CorpusRecord] = []
    if args.source in ("egdc", "all"):
        records.extend(egdc_records(variants=args.variants, seed=args.seed))
    if args.source in ("autoresearch", "all"):
        records.extend(
            autoresearch_records(solved_path=args.solved, queue_path=args.queue)
        )

    if args.only_verified:
        records = [r for r in records if r.verified]

    n = write_corpus(records, args.out)
    stats = summarize(records)
    print(json.dumps({"written": n, "out": str(args.out), **stats.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
