"""NPCoT-native coding self-consistency benchmark (BENCH-3).

**What this measures:** how faithfully a trained NPCoT library reproduces
ground truth on 200 array-reduction problems it was trained to handle.
That is, this is a *regression / self-consistency test for the library
itself*, not a comparison against real LLMs.

**What this is NOT:**
- NOT a comparison against Qwen / Gemma / any actual language model.
- NOT a HumanEval or MBPP replacement.
- NOT evidence of LLM improvement — for that use
  `humaneval_runner.py` / `mbpp_runner.py` which run real models.

The value of this benchmark is simple: if library-consult accuracy on
problems it was trained for drops below ~60% pass@1 (integer
ground truth, round-to-int scoring), then something regressed in the
library training / crystallization / execution pipeline. It is a
fast (<10 s) CPU-only signal we can gate CI on.

Use `ncpu.self_optimizing.humaneval_runner` and `mbpp_runner` for real
LLM-comparison benchmarks.

Usage::

    python3 -m benchmarks.benchmark_npcot_coding_bench
    python3 -m benchmarks.benchmark_npcot_coding_bench --n-problems 500 --seed 42
    python3 -m benchmarks.benchmark_npcot_coding_bench --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ncpu.self_optimizing.array_executable_thought_head import (
    ArrayExecutableThoughtHead,
    ArrayExecutableThoughtHeadConfig,
    _compute_operation_target,
    build_array_thought_smoke_batch,
    run_array_thought_smoke_train,
)
from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
)
from ncpu.self_optimizing.program_library_session import (
    ProgramLibrarySession,
    ProgramLibrarySessionConfig,
)


# ---------------------------------------------------------------------------
# Problem generator
# ---------------------------------------------------------------------------

# The core reduction skills we cover. These are chosen to be the ones that
# converge cleanly in M2 (see curriculum enrichment notes in MEMORY.md).
SUPPORTED_SKILLS: tuple[str, ...] = (
    "sum",
    "max",
    "min",
    "count_positive",
    "count_negative",
)


@dataclass
class CodingProblem:
    skill: str
    array: list[int]
    length: int
    ground_truth: float
    baseline_answer: float       # A plausible LLM-style off-by-one / wrong-op answer.
    prompt: str                  # Natural-language problem statement.


def _baseline_guess(skill: str, array: list[int], rng: random.Random) -> float:
    """Synthetic noise reference — *not* a real LLM.

    Produces deliberately-buggy answers (off-by-one, wrong-op, sign
    confusion) so that library-consult accuracy has something to regress
    against. This function does NOT represent an actual language model's
    performance — for real LLM pass@1, see `humaneval_runner` / `mbpp_runner`.

    Kept in the benchmark only as a CI canary: if NPCoT library accuracy
    drops below this synthetic noise floor, the library is broken.
    """
    if not array:
        return 0.0
    mode = rng.random()
    ground_truth = _compute_operation_target(skill, torch.tensor(array, dtype=torch.float32))
    if mode < 0.35:
        # Off-by-one error on the count-style answers.
        if skill in ("count_positive", "count_negative"):
            return ground_truth + rng.choice([-1.0, 1.0])
        return ground_truth + rng.choice([-1.0, 1.0])
    if mode < 0.6:
        # Wrong-op error — swap with a neighbor.
        wrong_skills = {
            "sum": "count_positive",
            "count_positive": "count_negative",
            "count_negative": "count_positive",
            "max": "min",
            "min": "max",
        }
        wrong = wrong_skills.get(skill, skill)
        if wrong == skill:
            return ground_truth
        return _compute_operation_target(wrong, torch.tensor(array, dtype=torch.float32))
    if mode < 0.85:
        # Sign confusion.
        return -ground_truth
    # Correct (LLMs do sometimes get this right — baseline is not 0% accurate).
    return ground_truth


def generate_problems(
    *,
    n_problems: int,
    seed: int,
    array_max_len: int = 6,
    value_low: int = -4,
    value_high: int = 4,
) -> list[CodingProblem]:
    rng = random.Random(seed)
    problems: list[CodingProblem] = []
    for index in range(n_problems):
        skill = SUPPORTED_SKILLS[index % len(SUPPORTED_SKILLS)]
        length = rng.randint(2, array_max_len)
        array = [rng.randint(value_low, value_high) for _ in range(length)]
        padded = array + [0] * (array_max_len - length)
        ground_truth = _compute_operation_target(
            skill, torch.tensor(array, dtype=torch.float32)
        )
        baseline = _baseline_guess(skill, array, rng)
        prompt = _skill_prompt(skill, padded, length)
        problems.append(
            CodingProblem(
                skill=skill,
                array=padded,
                length=length,
                ground_truth=float(ground_truth),
                baseline_answer=float(baseline),
                prompt=prompt,
            )
        )
    return problems


_SKILL_PROMPTS = {
    "sum": "Return the sum of the elements in `arr`.",
    "max": "Return the maximum of the elements in `arr`.",
    "min": "Return the minimum of the elements in `arr`.",
    "count_positive": "Return the count of strictly positive elements in `arr`.",
    "count_negative": "Return the count of strictly negative elements in `arr`.",
}


def _skill_prompt(skill: str, array: list[int], length: int) -> str:
    return f"{_SKILL_PROMPTS[skill]} (arr has length {length}: {array[:length]})"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class SystemResult:
    name: str
    exact_pass_count: int
    approximate_pass_count: int   # within |error| <= 0.5 (i.e. acceptable for rounding)
    total: int
    mean_abs_error: float
    total_elapsed_s: float

    @property
    def pass_at_1(self) -> float:
        return self.exact_pass_count / max(self.total, 1)

    @property
    def approximate_pass_rate(self) -> float:
        return self.approximate_pass_count / max(self.total, 1)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pass_at_1": self.pass_at_1,
            "approximate_pass_rate": self.approximate_pass_rate,
            "exact_pass_count": self.exact_pass_count,
            "approximate_pass_count": self.approximate_pass_count,
            "total": self.total,
            "mean_abs_error": self.mean_abs_error,
            "total_elapsed_s": self.total_elapsed_s,
        }


def evaluate_system(
    name: str,
    predictions: list[float | None],
    problems: list[CodingProblem],
    elapsed_s: float,
    *,
    exact_eps: float = 1e-3,
    approximate_eps: float = 0.5,
) -> SystemResult:
    # Exact-match criterion: round to nearest int (matches what an LLM would
    # emit as a token) and compare to integer ground truth. This is the
    # honest "did it get the right answer" question for integer-output
    # problems like sum / max / count.
    exact_count = 0
    approximate_count = 0
    total_err = 0.0
    total_err_count = 0
    for pred, problem in zip(predictions, problems):
        if pred is None or (isinstance(pred, float) and math.isnan(pred)):
            continue
        rounded_pred = round(pred)
        rounded_gt = round(problem.ground_truth)
        err = abs(pred - problem.ground_truth)
        if rounded_pred == rounded_gt:
            exact_count += 1
        if err < approximate_eps:
            approximate_count += 1
        total_err += err
        total_err_count += 1
    mae = total_err / max(total_err_count, 1) if total_err_count else float("inf")
    return SystemResult(
        name=name,
        exact_pass_count=exact_count,
        approximate_pass_count=approximate_count,
        total=len(problems),
        mean_abs_error=mae,
        total_elapsed_s=elapsed_s,
    )


# ---------------------------------------------------------------------------
# Systems under test
# ---------------------------------------------------------------------------


def run_baseline(problems: list[CodingProblem]) -> tuple[list[float], float]:
    """LLM-baseline-style buggy answers (pre-computed in CodingProblem)."""
    start = time.perf_counter()
    answers = [p.baseline_answer for p in problems]
    elapsed = time.perf_counter() - start
    return answers, elapsed


def run_ground_truth(problems: list[CodingProblem]) -> tuple[list[float], float]:
    """Reference implementation — always correct, sanity check."""
    start = time.perf_counter()
    answers = [p.ground_truth for p in problems]
    elapsed = time.perf_counter() - start
    return answers, elapsed


def run_npcot(
    problems: list[CodingProblem],
    library_path: Path,
    *,
    hidden_dim: int = 16,
    array_max_len: int = 6,
) -> tuple[list[float | None], float, float]:
    """Consult pre-trained library. Returns (answers, hit_rate, elapsed)."""
    library = ArrayProgramLibrary.load(library_path)
    # Build hidden-state prototypes for each supported skill using the same
    # `_operation_hidden_prototypes` the smoke batch does. Each problem's
    # hidden state is the prototype for its skill.
    from ncpu.self_optimizing.array_executable_thought_head import (
        _operation_hidden_prototypes,
    )

    prototypes = _operation_hidden_prototypes(hidden_dim, len(SUPPORTED_SKILLS))
    skill_to_index = {s: i for i, s in enumerate(SUPPORTED_SKILLS)}

    answers: list[float | None] = []
    hits = 0
    start = time.perf_counter()
    for problem in problems:
        prototype = prototypes[skill_to_index[problem.skill]]
        entry = library.lookup(prototype)
        if entry is None:
            answers.append(None)
            continue
        hits += 1
        arr_tensor = torch.tensor([problem.array], dtype=torch.float32)
        len_tensor = torch.tensor([float(problem.length)], dtype=torch.float32)
        result = entry.program.execute(arr_tensor, len_tensor).item()
        answers.append(result)
    elapsed = time.perf_counter() - start
    hit_rate = hits / max(len(problems), 1)
    return answers, hit_rate, elapsed


def train_library(
    *,
    library_path: Path,
    hidden_dim: int = 16,
    array_max_len: int = 6,
    samples_per_op: int = 12,
    steps: int = 500,
    seed: int = 0,
) -> ArrayProgramLibrary:
    """One-shot training of a reference library covering all 5 skills."""
    torch.manual_seed(seed)
    head = ArrayExecutableThoughtHead(
        ArrayExecutableThoughtHeadConfig(
            hidden_dim=hidden_dim, array_max_len=array_max_len,
        )
    )
    hidden, arrays, lengths, targets, _ = build_array_thought_smoke_batch(
        hidden_dim=hidden_dim,
        array_max_len=array_max_len,
        samples_per_op=samples_per_op,
        seed=seed,
        operations=SUPPORTED_SKILLS,
    )
    run_array_thought_smoke_train(
        head,
        hidden_state=hidden,
        array_inputs=arrays,
        lengths=lengths,
        targets=targets,
        steps=steps,
        learning_rate=5e-2,
    )
    if library_path.exists():
        library_path.unlink()
    session = ProgramLibrarySession(
        ProgramLibrarySessionConfig(
            library_path=library_path,
            convergence_gap_threshold=2.0,
        )
    )
    session.begin_task("bench3_training")
    session.apply_converged_program(
        head, hidden, arrays, lengths=lengths, temperature=0.05
    )
    session.end_task()
    return ArrayProgramLibrary.load(library_path)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_benchmark(
    *,
    n_problems: int = 200,
    seed: int = 0,
    library_path: Path = Path("/tmp/npcot_bench3_library.json"),
) -> dict:
    print("=" * 72)
    print(f"NPCoT-Native Coding Benchmark — {n_problems} problems")
    print("=" * 72)
    print()

    print(">>> [1/3] Training reference library (24 examples, 500 steps)")
    library = train_library(library_path=library_path, seed=seed)
    print(f"    Library: {len(library)} entries, {library_path.stat().st_size} bytes on disk")
    print()

    print(f">>> [2/3] Generating {n_problems} coding problems")
    problems = generate_problems(n_problems=n_problems, seed=seed + 1000)
    print(f"    Distribution: {sum(1 for p in problems if p.skill == 'sum')} sum, "
          f"{sum(1 for p in problems if p.skill == 'max')} max, "
          f"{sum(1 for p in problems if p.skill == 'min')} min, "
          f"{sum(1 for p in problems if p.skill == 'count_positive')} count+, "
          f"{sum(1 for p in problems if p.skill == 'count_negative')} count-")
    print()

    print(">>> [3/3] Running three systems")
    print()

    gt_answers, gt_elapsed = run_ground_truth(problems)
    gt_result = evaluate_system("Ground truth (reference)", gt_answers, problems, gt_elapsed)

    bl_answers, bl_elapsed = run_baseline(problems)
    bl_result = evaluate_system("synthetic noise reference (NOT a real LLM)", bl_answers, problems, bl_elapsed)

    npcot_answers, hit_rate, npcot_elapsed = run_npcot(problems, library_path)
    np_result = evaluate_system("NPCoT library", npcot_answers, problems, npcot_elapsed)

    print("     System                          pass@1    approx   MAE       time")
    print("     " + "-" * 70)
    for r in (gt_result, bl_result, np_result):
        print(
            f"     {r.name:32s} {r.pass_at_1 * 100:5.1f}%   "
            f"{r.approximate_pass_rate * 100:5.1f}%   "
            f"{r.mean_abs_error:7.3f}   {r.total_elapsed_s * 1000:6.2f} ms"
        )
    print()
    print(f"     NPCoT library hit rate: {hit_rate * 100:.1f}%")
    print()

    delta = np_result.pass_at_1 - bl_result.pass_at_1
    print(f"     NPCoT vs baseline: {delta * 100:+.1f} pp (pass@1)")
    print()

    report = {
        "n_problems": n_problems,
        "seed": seed,
        "library_path": str(library_path),
        "library_entries": len(library),
        "library_bytes": library_path.stat().st_size,
        "hit_rate": hit_rate,
        "results": {
            "ground_truth": gt_result.to_dict(),
            "llm_baseline_emulated": bl_result.to_dict(),
            "npcot_library": np_result.to_dict(),
        },
        "pass_at_1_delta": delta,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-problems", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--library-path", type=Path, default=Path("/tmp/npcot_bench3_library.json"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_benchmark(
        n_problems=args.n_problems,
        seed=args.seed,
        library_path=args.library_path,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
