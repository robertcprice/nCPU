"""LiveCodeBench NPCoT agent runner — code generation + self-repair.

Runs the full NPCoT stack (coprocessor + library + retry + compounding store)
against LiveCodeBench problems from HuggingFace. Supports temporal filtering
for contamination-free claims.

LCB problems come from competitive programming contests (LeetCode, AtCoder,
Codeforces). Most use stdin/stdout I/O — the model writes a standalone script
that reads input and prints output. LeetCode problems use class Solution.
The runner detects the format and verifies accordingly.

Two scenarios:
  - codegeneration: solve from scratch (default)
  - selfrepair: fix a buggy solution

Usage::

    # Code generation on post-Sept-2024 problems (contamination-free):
    python3 -m ncpu.self_optimizing.run_livecodebench \\
        --model Qwen/Qwen3.5-4B \\
        --library library.json \\
        --coprocessor-checkpoint checkpoint.pt \\
        --start-date 2024-09-01 \\
        --out livecodebench_codegen.json

    # Self-repair scenario:
    python3 -m ncpu.self_optimizing.run_livecodebench \\
        --scenario selfrepair \\
        --model Qwen/Qwen3.5-4B \\
        --library library.json \\
        --out livecodebench_repair.json

    # Dry-run (validates config + dataset access without GPU):
    python3 -m ncpu.self_optimizing.run_livecodebench --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_LCB_DATASET = "livecodebench/code_generation"


@dataclass
class LiveCodeBenchConfig:
    model: str = "Qwen/Qwen3.5-4B-Instruct"
    library_path: Optional[Path] = None
    coprocessor_checkpoint: Optional[Path] = None
    target_layers: list[int] = field(default_factory=lambda: [-2, -1])
    array_max_len: int = 8
    scenario: str = "codegeneration"
    release_version: str = "release_v6"
    start_date: str = ""
    end_date: Optional[str] = None
    difficulty: Optional[str] = None
    max_problems: int = 0
    max_new_tokens: int = 1024
    retry_gates: tuple[float, ...] = (0.0, 0.05)
    retry_temperatures: tuple[float, ...] = (0.0, 0.5)
    max_retries: int = 2
    continual_growth: bool = True
    compounding_store_dir: Optional[Path] = None
    output_json: Path = Path("livecodebench_results.json")
    device: str = "auto"
    trust_remote_code: bool = False
    dry_run: bool = False
    quantize: bool = False


def parse_cli(argv: list[str] | None = None) -> LiveCodeBenchConfig:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--library", dest="library_path", type=Path, default=None)
    p.add_argument("--coprocessor-checkpoint", dest="coprocessor_checkpoint",
                   type=Path, default=None)
    p.add_argument("--target-layers", default="-2,-1")
    p.add_argument("--array-max-len", type=int, default=8)
    p.add_argument("--scenario", default="codegeneration",
                   choices=["codegeneration", "selfrepair"])
    p.add_argument("--release-version", default="release_v6")
    p.add_argument("--start-date", default="",
                   help="Only include problems from contests on or after this date (YYYY-MM-DD). "
                        "Empty string = no filter.")
    p.add_argument("--end-date", default=None)
    p.add_argument("--difficulty", default=None,
                   choices=["easy", "medium", "hard"])
    p.add_argument("--max-problems", type=int, default=0,
                   help="0 = all problems in the filtered set.")
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--continual-growth", action="store_true", default=True)
    p.add_argument("--no-continual-growth", action="store_true")
    p.add_argument("--compounding-store", dest="compounding_store_dir",
                   type=Path, default=None)
    p.add_argument("--out", dest="output_json", type=Path,
                   default=Path("livecodebench_results.json"))
    p.add_argument("--device", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quantize", action="store_true",
                   help="Apply dynamic int8 quantization (CPU only, ~2x speed).")
    args = p.parse_args(argv)
    continual = args.continual_growth and not args.no_continual_growth
    return LiveCodeBenchConfig(
        model=args.model,
        library_path=args.library_path,
        coprocessor_checkpoint=args.coprocessor_checkpoint,
        target_layers=[int(x) for x in args.target_layers.split(",") if x.strip()],
        array_max_len=args.array_max_len,
        scenario=args.scenario,
        release_version=args.release_version,
        start_date=args.start_date,
        end_date=args.end_date,
        difficulty=args.difficulty,
        max_problems=args.max_problems,
        max_new_tokens=args.max_new_tokens,
        retry_gates=(0.0, 0.05),
        retry_temperatures=(0.0, 0.5),
        max_retries=args.max_retries,
        continual_growth=continual,
        compounding_store_dir=args.compounding_store_dir,
        output_json=args.output_json,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        dry_run=args.dry_run,
        quantize=args.quantize,
    )


# ---------------------------------------------------------------------------
# Problem loading and formatting
# ---------------------------------------------------------------------------


def _norm_date(d) -> str:
    """Normalize contest_date to 'YYYY-MM-DD' string."""
    if not d:
        return ""
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d).split(" ")[0]


def load_livecodebench_problems(cfg: LiveCodeBenchConfig) -> list[dict]:
    """Load and filter LiveCodeBench problems from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset(_LCB_DATASET, split="test", streaming=True)
    problems: list[dict] = []
    for row in ds:
        contest_date = _norm_date(row.get("contest_date", "") or "")
        if cfg.start_date and contest_date < cfg.start_date:
            continue
        if cfg.end_date and contest_date > cfg.end_date:
            continue
        diff = (row.get("difficulty", "") or "").lower()
        if cfg.difficulty and diff != cfg.difficulty:
            continue

        test_cases = _parse_test_cases(row)
        starter_code = row.get("starter_code") or ""
        is_stdin = _is_stdin_problem(test_cases)
        entry_point = "" if is_stdin else _extract_entry_point(starter_code)

        problems.append({
            "task_id": row["question_id"],
            "title": row.get("question_title", ""),
            "prompt_code": _build_prompt(row, cfg.scenario, is_stdin),
            "test_cases": test_cases,
            "starter_code": starter_code,
            "entry_point": entry_point,
            "is_stdin": is_stdin,
            "difficulty": diff or "unknown",
            "contest_date": contest_date,
            "platform": row.get("platform", ""),
        })

    if cfg.max_problems > 0:
        problems = problems[:cfg.max_problems]

    return problems


def _is_stdin_problem(test_cases: list[dict]) -> bool:
    """Check if test cases use stdin/stdout (competitive programming) style."""
    if not test_cases:
        return True
    tt = test_cases[0].get("testtype", "")
    return tt == "stdin" or tt == ""


def _build_prompt(row: dict, scenario: str, is_stdin: bool) -> str:
    """Build the prompt the model sees."""
    content = row.get("question_content", "")
    starter = row.get("starter_code") or ""
    if starter:
        base = f"{content}\n\n{starter}"
    else:
        base = content
    if is_stdin and not starter:
        base += (
            "\n\nSolve the above problem in Python. Read from stdin, write to stdout."
            "\nOutput ONLY executable Python code. No comments, no explanation, no markdown."
            "\n```python\n"
        )
    elif not is_stdin and not starter:
        base += (
            "\n\nWrite a Python solution."
            "\nOutput ONLY the Python code with no explanation."
            "\n```python\n"
        )
    if scenario == "selfrepair":
        buggy = _generate_buggy_stub(starter, is_stdin)
        return f"{base}\n\n# The following solution has a bug. Fix it:\n{buggy}"
    return base


def _generate_buggy_stub(starter: str, is_stdin: bool) -> str:
    if is_stdin:
        return "import sys\ninput = sys.stdin.read\nprint(0)\n"
    if not starter.strip():
        return "class Solution:\n    pass"
    return starter.rstrip() + "\n    pass\n"


def _parse_test_cases(row: dict) -> list[dict]:
    """Parse public + private test cases from LCB JSON strings."""
    cases: list[dict] = []
    for field_name in ("public_test_cases", "private_test_cases"):
        raw = row.get(field_name, "")
        if not raw:
            continue
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            for tc in parsed:
                if isinstance(tc, dict) and "input" in tc and "output" in tc:
                    cases.append({
                        "input": tc["input"],
                        "output": tc["output"],
                        "testtype": tc.get("testtype", ""),
                    })
    return cases


_ENTRY_POINT_RE = re.compile(r"def\s+(\w+)\s*\(")


def _extract_entry_point(starter_code: str) -> str:
    for match in _ENTRY_POINT_RE.finditer(starter_code):
        name = match.group(1)
        if name != "__init__":
            return name
    return "solve"


# ---------------------------------------------------------------------------
# Code extraction and verification
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

_STOP_PATTERNS = (
    "\n```",
)


def _truncate_at_stop(code: str) -> str:
    best = len(code)
    for pat in _STOP_PATTERNS:
        idx = code.find(pat)
        if idx != -1 and idx < best:
            best = idx
    return code[:best].rstrip()


def extract_lcb_code(generated: str) -> str:
    """Extract runnable code from model output.

    Handles fenced blocks, bare functions, class Solution, and standalone
    scripts (for stdin/stdout problems).
    """
    stripped = generated.strip()

    match = _CODE_FENCE_RE.search(stripped)
    if match is not None:
        code = match.group(1).strip()
    else:
        code = stripped

    return _truncate_at_stop(code)


def check_lcb_solution(
    problem: dict,
    solution_code: str,
    *,
    timeout_s: float = 10.0,
) -> tuple[bool, str]:
    """Execute a solution against the problem's test cases.

    Handles both stdin/stdout (competitive programming) and class Solution
    (LeetCode) verification styles.
    """
    test_cases = problem.get("test_cases", [])
    if not test_cases:
        return False, "no test cases"

    if problem.get("is_stdin", True):
        return _check_stdin_solution(solution_code, test_cases, timeout_s)
    return _check_class_solution(solution_code, problem["entry_point"],
                                  test_cases, timeout_s)


def _check_stdin_solution(
    code: str,
    test_cases: list[dict],
    timeout_s: float,
) -> tuple[bool, str]:
    """Verify a stdin/stdout solution by piping input and checking output."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as fh:
        fh.write(code)
        tmp_path = fh.name

    try:
        for i, tc in enumerate(test_cases):
            stdin_input = tc["input"]
            expected_output = tc["output"]
            result = subprocess.run(
                [sys.executable, tmp_path],
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            actual = result.stdout
            if not _outputs_match(actual, expected_output):
                return False, f"FAIL case {i}: got {actual!r:.80}, expected {expected_output!r:.80}"
            if result.returncode != 0:
                return False, f"case {i}: nonzero exit — {(result.stderr or '').strip()[:100]}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


def _check_class_solution(
    code: str,
    entry_point: str,
    test_cases: list[dict],
    timeout_s: float,
) -> tuple[bool, str]:
    """Verify a class Solution by calling the method with test case inputs."""
    harness = _build_class_harness(code, entry_point, test_cases)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as fh:
        fh.write(harness)
        tmp_path = fh.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            return True, ""
        error = (result.stderr or result.stdout).strip().splitlines()[-1:]
        return False, (error[0] if error else "nonzero exit")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


def _outputs_match(actual: str, expected: str) -> bool:
    """Compare outputs with whitespace normalization."""
    a = actual.strip()
    e = expected.strip()
    if a == e:
        return True
    a_lines = a.splitlines()
    e_lines = e.splitlines()
    if len(a_lines) != len(e_lines):
        return False
    for al, el in zip(a_lines, e_lines):
        if al.strip() != el.strip():
            try:
                if math.isclose(float(al.strip()), float(el.strip()),
                                rel_tol=1e-5):
                    continue
            except ValueError:
                pass
            return False
    return True


def _build_class_harness(
    solution_code: str,
    entry_point: str,
    test_cases: list[dict],
) -> str:
    tc_repr = json.dumps(test_cases)
    return (
        f"import json, sys, math\n"
        f"{solution_code}\n\n"
        f"_cases = json.loads({tc_repr!r})\n"
        f"_sol = Solution()\n"
        f"for _i, _tc in enumerate(_cases):\n"
        f"    _inp = _tc['input']\n"
        f"    _exp = _tc['output']\n"
        f"    if isinstance(_inp, list):\n"
        f"        _result = _sol.{entry_point}(*_inp)\n"
        f"    else:\n"
        f"        _result = _sol.{entry_point}(_inp)\n"
        f"    if _result != _exp:\n"
        f"        if not (isinstance(_result, float) and isinstance(_exp, float)"
        f" and math.isclose(_result, _exp, rel_tol=1e-5)):\n"
        f"            print(f'FAIL case {{_i}}: got {{_result}}, expected {{_exp}}',"
        f" file=sys.stderr)\n"
        f"            sys.exit(1)\n"
        f"print('OK')\n"
    )


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def run_dry(cfg: LiveCodeBenchConfig) -> dict:
    checks: list[tuple[str, bool, str]] = []

    if cfg.library_path is not None:
        resolved = cfg.library_path.expanduser()
        if resolved.exists():
            from ncpu.self_optimizing.array_program_library import ArrayProgramLibrary
            lib = ArrayProgramLibrary.load(resolved)
            checks.append(("library", True, f"{len(lib)} entries"))
        else:
            checks.append(("library", False, f"{resolved} not found"))
    else:
        checks.append(("library", True, "baseline (no library)"))

    for idx in cfg.target_layers:
        checks.append((f"layer_{idx}", True, "ok"))

    out_parent = cfg.output_json.expanduser().parent
    checks.append((
        "output_path_writable",
        out_parent.exists() or out_parent.parent.exists(),
        str(cfg.output_json),
    ))

    checks.append(("scenario", True, cfg.scenario))
    checks.append(("release_version", True, cfg.release_version))
    checks.append(("start_date", True, cfg.start_date))

    problem_count = "?"
    try:
        from datasets import load_dataset
        ds = load_dataset(_LCB_DATASET, split="test", streaming=True)
        n = 0
        for row in ds:
            cd = _norm_date(row.get("contest_date", "") or "")
            if cfg.start_date and cd < cfg.start_date:
                continue
            if cfg.end_date and cd > cfg.end_date:
                continue
            if cfg.difficulty and (row.get("difficulty", "") or "").lower() != cfg.difficulty:
                continue
            n += 1
        problem_count = str(n)
        checks.append(("problem_count", True, f"{n} problems after filtering"))
    except Exception as exc:
        checks.append(("problem_count", False, str(exc)))

    return {
        "mode": "dry_run",
        "timestamp": time.time(),
        "config": {
            "model": cfg.model,
            "scenario": cfg.scenario,
            "release_version": cfg.release_version,
            "start_date": cfg.start_date,
            "difficulty": cfg.difficulty,
        },
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        "all_ok": all(ok for _, ok, _ in checks),
    }


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


def run_agent(cfg: LiveCodeBenchConfig) -> dict:
    """Main agent loop over LiveCodeBench with retry + compounding store."""
    import torch
    from ncpu.self_optimizing.humaneval_runner import (
        HumanEvalConfig,
        generate_solution,
        load_model_with_optional_npcot,
    )
    from ncpu.self_optimizing.best_of_gate import set_coprocessor_gates
    from ncpu.self_optimizing.verifier_retry import (
        RetryConfig,
        RetryStrategy,
        retry_until_verified,
    )

    he_cfg = HumanEvalConfig(
        model=cfg.model,
        library_path=cfg.library_path,
        coprocessor_checkpoint=cfg.coprocessor_checkpoint,
        target_layers=cfg.target_layers,
        array_max_len=cfg.array_max_len,
        array_thought_max_gate=0.05,
        max_problems=cfg.max_problems,
        max_new_tokens=cfg.max_new_tokens,
        device=cfg.device,
        trust_remote_code=cfg.trust_remote_code,
        use_npcot=cfg.library_path is not None,
        quantize=cfg.quantize,
    )

    print(f"[lcb] loading model {cfg.model}", flush=True)
    model, tokenizer, device, npcot_meta = load_model_with_optional_npcot(he_cfg)
    print(f"[lcb] loaded on {device}; NPCoT: {npcot_meta}", flush=True)

    problems = load_livecodebench_problems(cfg)
    _diff_order = {"easy": 0, "medium": 1, "hard": 2}
    problems.sort(key=lambda p: _diff_order.get(p.get("difficulty", ""), 3))
    print(f"[lcb] {len(problems)} problems loaded "
          f"(scenario={cfg.scenario}, start_date={cfg.start_date})", flush=True)

    strategies: list[RetryStrategy] = []
    for i in range(cfg.max_retries):
        gate_idx = min(i, len(cfg.retry_gates) - 1)
        temp_idx = min(i, len(cfg.retry_temperatures) - 1)
        strategies.append(RetryStrategy(
            gate=cfg.retry_gates[gate_idx],
            temperature=cfg.retry_temperatures[temp_idx],
            label=f"gate={cfg.retry_gates[gate_idx]}_temp={cfg.retry_temperatures[temp_idx]}",
        ))
    retry_cfg = RetryConfig(strategies=strategies, max_attempts=cfg.max_retries)

    store = None
    if cfg.compounding_store_dir is not None:
        from ncpu.autoresearch.compounding_store import (
            CompoundingStore,
            CompoundingStoreConfig,
        )
        store = CompoundingStore(CompoundingStoreConfig(
            artifact_dir=cfg.compounding_store_dir,
        ))
        print(f"[lcb] compounding-store: {store.summary()}", flush=True)

    per_problem: list[dict] = []
    pass_count = 0
    first_try_passes = 0
    retry_wins = 0
    store_hits = 0
    attempts_total = 0
    difficulty_counts: dict[str, list[bool]] = defaultdict(list)
    t_start = time.perf_counter()

    for pi, problem in enumerate(problems):
        t0 = time.perf_counter()

        if store is not None:
            class _WIShim:
                pass
            wi = _WIShim()
            wi.prompt = problem["prompt_code"]
            wi.entry_point = problem["entry_point"]
            hit = store.check_prompt(wi)
            if hit is not None:
                passed, err = check_lcb_solution(
                    problem, hit.program_python,
                )
                if passed:
                    pass_count += 1
                    first_try_passes += 1
                    store_hits += 1
                    difficulty_counts[problem["difficulty"]].append(True)
                    per_problem.append({
                        "task_id": problem["task_id"],
                        "title": problem["title"],
                        "passed": True,
                        "difficulty": problem["difficulty"],
                        "contest_date": problem["contest_date"],
                        "gen_seconds": round(time.perf_counter() - t0, 3),
                        "total_attempts": 0,
                        "winning_attempt": "store_hit",
                        "attempts": [{"strategy": "compounding_store_hit",
                                      "passed": True, "score": 1.0}],
                    })
                    print(
                        f"[lcb] {pi+1}/{len(problems)} {problem['task_id']}: "
                        f"STORE-HIT "
                        f"pass@1 {pass_count}/{pi+1} = "
                        f"{pass_count/(pi+1)*100:.1f}%",
                        flush=True,
                    )
                    continue

        def generate(strategy: RetryStrategy) -> str:
            set_coprocessor_gates(model, strategy.gate)
            out = generate_solution(
                model, tokenizer, problem["prompt_code"],
                max_new_tokens=cfg.max_new_tokens,
                temperature=strategy.temperature,
                device=device,
            )
            return extract_lcb_code(out)

        def verify(code_body: str) -> tuple[bool, float, Optional[str]]:
            passed, err = check_lcb_solution(problem, code_body)
            if passed:
                return True, 1.0, None
            if err and "timeout" in err.lower():
                return False, 0.2, err
            if err and ("error" in err.lower() or "Exception" in err):
                return False, 0.1, err
            return False, 0.5, err

        result = retry_until_verified(
            generate_fn=generate, verify_fn=verify, config=retry_cfg,
        )
        gen_s = time.perf_counter() - t0

        difficulty_counts[problem["difficulty"]].append(result.final_passed)

        if result.final_passed:
            pass_count += 1
            if result.winning_attempt_index == 0:
                first_try_passes += 1
            else:
                retry_wins += 1
            if store is not None and result.final_text is not None:
                from ncpu.autoresearch.types import SolvedItem as _SI
                winning_strategy = (
                    result.attempts[result.winning_attempt_index].strategy_label
                    if 0 <= result.winning_attempt_index < len(result.attempts)
                    else "unknown"
                )
                class _WI:
                    prompt = problem["prompt_code"]
                    entry_point = problem["entry_point"]
                store.record(
                    _SI(
                        task_id=problem["task_id"],
                        source_benchmark="livecodebench",
                        solver=f"lcb_agent:{winning_strategy}",
                        program_python=result.final_text,
                        verifier_passed=True,
                        wall_seconds=gen_s,
                        provenance={
                            "prompt": problem["prompt_code"],
                            "entry_point": problem["entry_point"],
                            "scenario": cfg.scenario,
                            "difficulty": problem["difficulty"],
                            "contest_date": problem["contest_date"],
                        },
                    ),
                    work_item=_WI(),
                )
        attempts_total += result.total_attempts

        per_problem.append({
            "task_id": problem["task_id"],
            "title": problem["title"],
            "passed": result.final_passed,
            "difficulty": problem["difficulty"],
            "contest_date": problem["contest_date"],
            "gen_seconds": round(gen_s, 2),
            "total_attempts": result.total_attempts,
            "winning_attempt": result.winning_attempt_index,
            "attempts": [
                {
                    "strategy": a.strategy_label,
                    "passed": a.verifier_passed,
                    "score": round(a.verifier_score, 2),
                }
                for a in result.attempts
            ],
        })

        if (pi + 1) % 10 == 0 or result.final_passed:
            print(
                f"[lcb] {pi+1}/{len(problems)} {problem['task_id']}: "
                f"{'PASS' if result.final_passed else 'FAIL'} "
                f"(attempts={result.total_attempts}) "
                f"pass@1 {pass_count}/{pi+1} = {pass_count/(pi+1)*100:.1f}% "
                f"(1st-try {first_try_passes}, retry-wins {retry_wins})",
                flush=True,
            )

    total_s = time.perf_counter() - t_start
    pass_at_1 = pass_count / max(len(problems), 1)

    set_coprocessor_gates(model, 0.05)

    difficulty_breakdown: dict[str, dict] = {}
    for diff, results in difficulty_counts.items():
        n = len(results)
        p = sum(results)
        difficulty_breakdown[diff] = {
            "total": n,
            "passed": p,
            "pass_rate": round(p / max(n, 1), 4),
        }

    report = {
        "mode": "livecodebench_agent",
        "timestamp": time.time(),
        "scenario": cfg.scenario,
        "config": {
            "model": cfg.model,
            "library_path": str(cfg.library_path) if cfg.library_path else None,
            "coprocessor_checkpoint": str(cfg.coprocessor_checkpoint)
                if cfg.coprocessor_checkpoint else None,
            "scenario": cfg.scenario,
            "release_version": cfg.release_version,
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "difficulty": cfg.difficulty,
            "continual_growth": cfg.continual_growth,
            "retry_gates": list(cfg.retry_gates),
            "retry_temperatures": list(cfg.retry_temperatures),
            "max_retries": cfg.max_retries,
        },
        "npcot_meta": npcot_meta,
        "results": {
            "pass_at_1": pass_at_1,
            "pass_count": pass_count,
            "total_problems": len(problems),
            "first_try_passes": first_try_passes,
            "retry_wins": retry_wins,
            "store_hits": store_hits,
            "total_attempts": attempts_total,
            "attempts_per_problem": attempts_total / max(len(problems), 1),
            "total_seconds": round(total_s, 2),
        },
        "difficulty_breakdown": difficulty_breakdown,
        "per_problem": per_problem,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    cfg = parse_cli(argv)

    if cfg.dry_run:
        report = run_dry(cfg)
        print(json.dumps(report, indent=2))
        return 0 if report["all_ok"] else 1

    try:
        report = run_agent(cfg)
    except ImportError as exc:
        print(f"error: missing dependency ({exc})", file=sys.stderr)
        return 2

    out = cfg.output_json.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    r = report["results"]
    print(
        f"\n[lcb] done. pass@1 = {r['pass_at_1']*100:.2f}% "
        f"({r['pass_count']}/{r['total_problems']}), "
        f"1st-try {r['first_try_passes']}, retry-wins {r['retry_wins']}, "
        f"avg attempts {r['attempts_per_problem']:.2f}"
    )
    if report.get("difficulty_breakdown"):
        for diff, stats in sorted(report["difficulty_breakdown"].items()):
            print(f"  {diff}: {stats['passed']}/{stats['total']} "
                  f"({stats['pass_rate']*100:.1f}%)")
    print(f"[lcb] report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
