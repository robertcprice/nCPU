"""MBPP evaluator harness (BENCH-2).

Same skeleton as `humaneval_runner` but for Mostly Basic Python Problems
(MBPP). MBPP has 974 problems of roughly HumanEval-level difficulty but
different problem style: each entry gives a natural-language description
plus test cases, and the model is expected to produce a function that
passes the tests.

The MBPP problem style (many filter / count / reduce tasks) is a better
match for NPCoT's array-reduction sweet spot than HumanEval, so this is
the benchmark where we expect the clearest library-enabled delta.

Usage::

    python3 -m ncpu.self_optimizing.mbpp_runner --dry-run
    python3 -m ncpu.self_optimizing.mbpp_runner \\
        --model Qwen/Qwen3.5-1.5B \\
        --library ~/.nCPU_program_library.json \\
        --max-problems 100 \\
        --out mbpp_run.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ncpu.self_optimizing.humaneval_runner import (
    _extract_code,
    load_model_with_optional_npcot,
    generate_solution,
)


@dataclass
class MBPPConfig:
    model: str = "Qwen/Qwen3.5-1.5B"
    library_path: Optional[Path] = None
    target_layers: list[int] = field(default_factory=lambda: [-2, -1])
    array_max_len: int = 8
    array_thought_max_gate: float = 0.05
    max_problems: int = 100
    max_new_tokens: int = 400
    temperature: float = 0.0
    output_json: Path = Path("mbpp_run.json")
    dry_run: bool = False
    device: str = "auto"
    trust_remote_code: bool = False
    use_npcot: bool = True


def parse_cli(argv: list[str] | None = None) -> MBPPConfig:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3.5-1.5B")
    p.add_argument("--library", dest="library_path", type=Path, default=None)
    p.add_argument("--no-library", action="store_true")
    p.add_argument("--target-layers", default="-2,-1")
    p.add_argument("--array-max-len", type=int, default=8)
    p.add_argument("--array-thought-max-gate", type=float, default=0.05)
    p.add_argument("--max-problems", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", dest="output_json", type=Path, default=Path("mbpp_run.json"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args(argv)
    return MBPPConfig(
        model=args.model,
        library_path=args.library_path,
        target_layers=[int(x) for x in args.target_layers.split(",") if x.strip()],
        array_max_len=args.array_max_len,
        array_thought_max_gate=args.array_thought_max_gate,
        max_problems=args.max_problems,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_json=args.output_json,
        dry_run=args.dry_run,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        use_npcot=not args.no_library,
    )


def run_dry(cfg: MBPPConfig) -> dict:
    """Validate config without loading GPU deps."""
    checks = []
    if cfg.use_npcot:
        if cfg.library_path is None:
            checks.append(("library_path", False, "--library required when use_npcot"))
        else:
            resolved = cfg.library_path.expanduser()
            if not resolved.exists():
                checks.append(("library_exists", False, f"{resolved} not found"))
            else:
                from ncpu.self_optimizing.array_program_library import ArrayProgramLibrary
                lib = ArrayProgramLibrary.load(resolved)
                checks.append(("library_exists", True, f"{len(lib)} entries"))
    else:
        checks.append(("library_mode", True, "baseline"))
    return {
        "mode": "dry_run",
        "timestamp": time.time(),
        "config": {
            "model": cfg.model,
            "library_path": str(cfg.library_path) if cfg.library_path else None,
            "use_npcot": cfg.use_npcot,
            "max_problems": cfg.max_problems,
        },
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        "all_ok": all(ok for _, ok, _ in checks),
    }


def load_mbpp_problems(max_problems: int) -> list[dict]:
    """Load MBPP from HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset("mbpp", split="test")
    problems = []
    for i, row in enumerate(ds):
        if i >= max_problems:
            break
        problems.append({
            "task_id": f"mbpp/{row['task_id']}",
            "text": row["text"],
            "code": row["code"],
            "test_list": row["test_list"],
            "test_setup_code": row.get("test_setup_code", ""),
        })
    return problems


def _mbpp_prompt(problem: dict) -> str:
    """Build an LLM prompt from an MBPP row in the standard 3-shot style."""
    tests = "\n".join(problem["test_list"])
    return (
        f"You are an expert Python programmer, and here is your task: "
        f"{problem['text']} "
        f"Your code should pass these tests:\n\n{tests}\n\n[BEGIN]\n"
    )


def _check_mbpp(problem: dict, solution: str, *, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Execute MBPP tests against a solution."""
    parts = [
        problem.get("test_setup_code", ""),
        "# === solution ===",
        solution,
        "# === test ===",
    ]
    parts.extend(problem["test_list"])
    parts.append("print('OK')")
    harness = "\n".join(parts)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
        fh.write(harness)
        tmp_path = fh.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout_s,
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


def run_mbpp(cfg: MBPPConfig) -> dict:
    from ncpu.self_optimizing.compliance_report import (
        ComplianceReportConfig,
        build_compliance_report,
    )

    # Reuse humaneval_runner's loader for consistency.
    from ncpu.self_optimizing.humaneval_runner import HumanEvalConfig
    he_cfg = HumanEvalConfig(
        model=cfg.model,
        library_path=cfg.library_path,
        target_layers=cfg.target_layers,
        array_max_len=cfg.array_max_len,
        array_thought_max_gate=cfg.array_thought_max_gate,
        device=cfg.device,
        trust_remote_code=cfg.trust_remote_code,
        use_npcot=cfg.use_npcot,
    )
    print(f"[mbpp] loading model {cfg.model}", flush=True)
    model, tokenizer, device, npcot_meta = load_model_with_optional_npcot(he_cfg)
    print(f"[mbpp] loaded on {device}; NPCoT: {npcot_meta}", flush=True)

    print(f"[mbpp] loading up to {cfg.max_problems} problems", flush=True)
    problems = load_mbpp_problems(cfg.max_problems)
    print(f"[mbpp] {len(problems)} problems loaded", flush=True)

    per_problem = []
    pass_count = 0
    t_start = time.perf_counter()
    for i, problem in enumerate(problems):
        prompt = _mbpp_prompt(problem)
        t0 = time.perf_counter()
        generated = generate_solution(
            model, tokenizer, prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            device=device,
        )
        gen_s = time.perf_counter() - t0
        code = _extract_code(generated, prompt)
        # MBPP: the model emits a complete function definition.
        passed, err = _check_mbpp(problem, code)
        if passed:
            pass_count += 1
        per_problem.append({
            "task_id": problem["task_id"],
            "passed": passed,
            "gen_seconds": round(gen_s, 3),
            "code_chars": len(code),
            "error": err if not passed else None,
        })
        if (i + 1) % 10 == 0 or passed:
            print(
                f"[mbpp] {i+1}/{len(problems)} "
                f"{problem['task_id']}: {'PASS' if passed else 'FAIL'} "
                f"(pass@1 so far {pass_count}/{i+1} = {pass_count/(i+1)*100:.1f}%)",
                flush=True,
            )
    total_s = time.perf_counter() - t_start

    report = {
        "mode": "mbpp_real_run",
        "timestamp": time.time(),
        "config": {
            "model": cfg.model,
            "library_path": str(cfg.library_path) if cfg.library_path else None,
            "use_npcot": cfg.use_npcot,
            "target_layers": cfg.target_layers,
            "max_problems": cfg.max_problems,
        },
        "npcot_meta": npcot_meta,
        "results": {
            "pass_at_1": pass_count / max(len(problems), 1),
            "pass_count": pass_count,
            "total_problems": len(problems),
            "total_seconds": round(total_s, 2),
        },
        "per_problem": per_problem,
    }
    if cfg.use_npcot and cfg.library_path is not None:
        from ncpu.self_optimizing.array_program_library import ArrayProgramLibrary
        library = ArrayProgramLibrary.load(cfg.library_path.expanduser())
        report["compliance"] = build_compliance_report(
            library,
            config=ComplianceReportConfig(library_name=cfg.library_path.stem),
        )
    return report


def main(argv: list[str] | None = None) -> int:
    cfg = parse_cli(argv)
    if cfg.dry_run:
        report = run_dry(cfg)
        print(json.dumps(report, indent=2))
        return 0 if report["all_ok"] else 1
    try:
        report = run_mbpp(cfg)
    except ImportError as exc:
        print(f"error: missing dependency ({exc}).", file=sys.stderr)
        return 2
    out = cfg.output_json.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    r = report["results"]
    print(f"\n[mbpp] done. pass@1 = {r['pass_at_1'] * 100:.2f}% "
          f"({r['pass_count']}/{r['total_problems']}), wall {r['total_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
