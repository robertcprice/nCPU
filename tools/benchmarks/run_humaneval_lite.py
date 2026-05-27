#!/usr/bin/env python3
"""
HumanEval-lite runner for nsynth.

For each JSONL problem in the benchmark set:
  1. Call `nsynth_codegen --lang python` with the (name, signature, examples)
  2. Execute the generated Python against `test_cases`
  3. Record pass/fail + solve time + synthesis method

Output: artifacts/humaneval_results.md with per-problem rows + pass@1.
Exit code matches success rate so a CI can gate on it.

Usage:
    python3 tools/benchmarks/run_humaneval_lite.py \
        [--problems tools/benchmarks/humaneval_lite.jsonl] \
        [--out artifacts/humaneval_results.md] \
        [--timeout 20] \
        [--verbose]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Problem:
    name: str
    signature: str
    examples: List[dict]
    test_cases: List[list]  # list of [arg1, arg2, ..., expected]


@dataclass
class Result:
    name: str
    solved: bool
    pass_at_1: bool
    solve_ms: int
    method: str
    error: str = ""
    generated_code: str = ""
    failing_case: Optional[list] = None


def load_problems(path: Path) -> List[Problem]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(
            Problem(
                name=d["name"],
                signature=d["signature"],
                examples=d["examples"],
                test_cases=d["test_cases"],
            )
        )
    return out


def run_synthesis(
    codegen_bin: Path, problem: Problem, timeout: int
) -> tuple[str, int, str]:
    """Call nsynth_codegen, return (python_code, elapsed_ms, method_or_error)."""
    spec = json.dumps({
        "name": problem.name,
        "signature": problem.signature,
        "examples": problem.examples,
    })
    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(codegen_bin), "--lang", "python", "--examples", spec, "--verbose"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ("", int((time.time() - t0) * 1000), "TIMEOUT")
    elapsed_ms = int((time.time() - t0) * 1000)
    if proc.returncode != 0:
        return ("", elapsed_ms, proc.stderr.strip()[:300] or "nonzero-exit")
    # Extract method from stderr (emitted by --verbose).
    method = "unknown"
    for line in proc.stderr.splitlines():
        if "solved via" in line:
            method = line.split("solved via", 1)[1].split(" in ")[0].strip()
            break
    return (proc.stdout, elapsed_ms, method)


def verify(code: str, problem: Problem) -> tuple[bool, Optional[list], str]:
    """Execute the generated code, call the function on each test case,
    return (pass_at_1, failing_case_or_None, error_msg)."""
    ns: dict = {}
    try:
        exec(code, ns)
    except Exception as e:
        return (False, None, f"exec-error: {e!r}")
    fn = ns.get(problem.name)
    if fn is None:
        return (False, None, f"function {problem.name} not defined")
    for case in problem.test_cases:
        *args, expected = case
        try:
            got = fn(*args)
        except Exception as e:
            return (False, case, f"call-error: {e!r}")
        if got != expected:
            return (False, case, f"wrong: got {got}, expected {expected}")
    return (True, None, "")


def run_all(
    problems: List[Problem], codegen_bin: Path, timeout: int, verbose: bool
) -> List[Result]:
    results: List[Result] = []
    for i, p in enumerate(problems, 1):
        if verbose:
            print(f"[{i}/{len(problems)}] {p.name} ...", end=" ", flush=True)
        code, elapsed_ms, method_or_err = run_synthesis(codegen_bin, p, timeout)
        if not code:
            results.append(
                Result(
                    name=p.name,
                    solved=False,
                    pass_at_1=False,
                    solve_ms=elapsed_ms,
                    method=method_or_err,
                    error="synthesis failed",
                )
            )
            if verbose:
                print(f"MISS ({method_or_err})")
            continue
        passed, failing_case, err = verify(code, p)
        results.append(
            Result(
                name=p.name,
                solved=True,
                pass_at_1=passed,
                solve_ms=elapsed_ms,
                method=method_or_err,
                error=err,
                generated_code=code,
                failing_case=failing_case,
            )
        )
        if verbose:
            print("✓" if passed else f"✗ {err[:80]}")
    return results


def write_report(
    results: List[Result], out: Path, total_problems: int, total_ms: int
) -> None:
    solved = sum(1 for r in results if r.solved)
    passed = sum(1 for r in results if r.pass_at_1)
    pass_at_1_pct = 100.0 * passed / max(total_problems, 1)
    solve_pct = 100.0 * solved / max(total_problems, 1)

    lines: List[str] = []
    lines.append("# HumanEval-lite Results")
    lines.append("")
    lines.append(
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} — "
        f"{total_problems} problems, total runtime {total_ms/1000:.1f}s."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Synthesis success rate**: {solved}/{total_problems} ({solve_pct:.1f}%)")
    lines.append(
        f"- **Pass@1 (code runs + passes all test cases)**: **{passed}/{total_problems} ({pass_at_1_pct:.1f}%)**"
    )
    lines.append("")
    lines.append("## Per-problem results")
    lines.append("")
    lines.append("| # | problem | synth | pass@1 | ms | method | notes |")
    lines.append("|--:|---------|:-----:|:------:|---:|--------|-------|")
    for i, r in enumerate(results, 1):
        synth = "✓" if r.solved else "✗"
        p1 = "✓" if r.pass_at_1 else "✗"
        notes = ""
        if not r.solved:
            notes = f"synth: {r.method[:40]}"
        elif not r.pass_at_1:
            notes = f"{r.error[:60]}"
        lines.append(
            f"| {i} | {r.name} | {synth} | {p1} | {r.solve_ms} | {r.method[:30]} | {notes} |"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--problems",
        default="tools/benchmarks/humaneval_lite.jsonl",
    )
    ap.add_argument("--out", default="artifacts/humaneval_results.md")
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--codegen",
        default="nsynth/target/release/nsynth_codegen",
        help="Path to the nsynth_codegen binary",
    )
    args = ap.parse_args()

    # Resolve paths relative to the repo root. Running this script from
    # anywhere inside the repo should Just Work.
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    problems_path = Path(args.problems)
    out_path = Path(args.out)
    codegen_bin = Path(args.codegen)

    if not problems_path.exists():
        print(f"[runner] no problem set at {problems_path}", file=sys.stderr)
        sys.exit(2)
    if not codegen_bin.exists():
        print(
            f"[runner] nsynth_codegen not built at {codegen_bin} — "
            f"run `cargo build --release --bin nsynth_codegen` first",
            file=sys.stderr,
        )
        sys.exit(2)

    problems = load_problems(problems_path)
    print(f"[runner] {len(problems)} problems loaded")

    t0 = time.time()
    results = run_all(problems, codegen_bin, args.timeout, args.verbose)
    total_ms = int((time.time() - t0) * 1000)

    write_report(results, out_path, len(problems), total_ms)

    passed = sum(1 for r in results if r.pass_at_1)
    solved = sum(1 for r in results if r.solved)
    print(
        f"[runner] wrote {out_path} "
        f"(synth {solved}/{len(problems)}, pass@1 {passed}/{len(problems)}, "
        f"total {total_ms/1000:.1f}s)"
    )


if __name__ == "__main__":
    main()
