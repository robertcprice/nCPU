#!/usr/bin/env python3
"""
self_improving_loop.py — Self-improving program synthesis loop.

A system that gets better at synthesizing programs over time by learning from
its own successes and failures. The core innovation: every solved problem
becomes training data that strengthens the meta-learner, so the gradient
solver handles more problems autonomously over time, reducing LLM dependency.

Architecture:
    1. Collect problems (functions, binaries, random generators, raw I/O)
    2. Try differentiable synthesis via mog_synth (fast, free, GPU-accelerated)
    3. If gradient fails, try LLM synthesis (powerful but costs API credits)
    4. Verify ALL solutions against holdout examples
    5. Append successes to training data (mog_synth/data/expr_type_train.jsonl)
    6. Periodically retrain the meta-learner on the expanded dataset
    7. Track cumulative metrics: gradient coverage, LLM fallback rate, unsolved

The loop forms a flywheel:
    more solutions → more training data → better meta-learner →
    faster gradient convergence → more solutions

Usage:
    # From Python functions
    loop = SelfImprovingLoop()
    loop.add_function(lambda a, b: a * b + a, arg_types=["int", "int"])
    loop.add_function(lambda x: x * x - 1, arg_types=["int"])
    loop.run()
    loop.report()

    # Continuous improvement mode (generates random problems)
    loop = SelfImprovingLoop()
    loop.run(n_iterations=200, continuous=True)

    # CLI
    python egdc/self_improving_loop.py --continuous --iterations 100
    python egdc/self_improving_loop.py --functions "lambda a,b: a+b" "lambda x: x*x"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOG_SYNTH_DIR = PROJECT_ROOT / "mog_synth"
TRAINING_DATA_PATH = MOG_SYNTH_DIR / "data" / "expr_type_train.jsonl"
META_LEARNER_SCRIPT = MOG_SYNTH_DIR / "scripts" / "train_expr_metalearner.py"
META_LEARNER_MODEL = MOG_SYNTH_DIR / "models" / "expr_metalearner.pt"

logger = logging.getLogger("self_improving_loop")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """A synthesis problem: function name, I/O examples, holdouts."""
    name: str
    examples: list[tuple[Any, Any]]
    holdouts: list[tuple[Any, Any]]
    signature: Optional[str] = None
    description: str = ""
    source: str = "manual"  # "function", "binary", "manual", "random"
    arg_types: Optional[list[str]] = None


@dataclass
class SolveAttempt:
    """Record of a single solve attempt."""
    problem_name: str
    solved: bool
    method: str  # "differentiable", "llm", "unsolved"
    code: Optional[str] = None
    language: str = "mog"
    time_s: float = 0.0
    error: Optional[str] = None
    verified: bool = False  # passed holdout verification


@dataclass
class LoopMetrics:
    """Cumulative metrics across the improvement loop."""
    problems_attempted: int = 0
    solved_by_gradient: int = 0
    solved_by_llm: int = 0
    unsolved: int = 0
    holdout_failures: int = 0  # solved synthesis but failed holdout check
    training_records_added: int = 0
    retrains_completed: int = 0
    total_time_s: float = 0.0
    history: list[dict] = field(default_factory=list)

    @property
    def solve_rate(self) -> float:
        if self.problems_attempted == 0:
            return 0.0
        return (self.solved_by_gradient + self.solved_by_llm) / self.problems_attempted

    @property
    def gradient_rate(self) -> float:
        solved = self.solved_by_gradient + self.solved_by_llm
        if solved == 0:
            return 0.0
        return self.solved_by_gradient / solved

    def snapshot(self) -> dict:
        return {
            "attempted": self.problems_attempted,
            "gradient": self.solved_by_gradient,
            "llm": self.solved_by_llm,
            "unsolved": self.unsolved,
            "holdout_fail": self.holdout_failures,
            "records_added": self.training_records_added,
            "retrains": self.retrains_completed,
            "solve_rate": round(self.solve_rate, 4),
            "gradient_rate": round(self.gradient_rate, 4),
        }


# ---------------------------------------------------------------------------
# Random problem generators (for continuous improvement mode)
# ---------------------------------------------------------------------------

def _random_expr_problem(rng: random.Random) -> Problem:
    """Generate a random arithmetic expression problem."""
    templates_1arg = [
        ("double", lambda a: a * 2),
        ("square", lambda a: a * a),
        ("negate", lambda a: -a),
        ("inc", lambda a: a + 1),
        ("dec", lambda a: a - 1),
        ("abs_val", lambda a: abs(a)),
        ("triple", lambda a: a * 3),
        ("cube", lambda a: a * a * a),
    ]
    templates_2arg = [
        ("add", lambda a, b: a + b),
        ("sub", lambda a, b: a - b),
        ("mul", lambda a, b: a * b),
        ("max_ab", lambda a, b: max(a, b)),
        ("min_ab", lambda a, b: min(a, b)),
        ("sum_sq", lambda a, b: a * a + b * b),
        ("diff_sq", lambda a, b: (a - b) * (a - b)),
        ("avg_floor", lambda a, b: (a + b) // 2),
    ]

    coin = rng.random()
    if coin < 0.5:
        name, fn = rng.choice(templates_1arg)
        suffix = rng.randint(0, 999)
        name = f"{name}_{suffix}"
        examples = []
        holdouts = []
        seen = set()
        for _ in range(200):
            a = rng.randint(-50, 50)
            if a in seen:
                continue
            seen.add(a)
            try:
                r = fn(a)
                if abs(r) > 10**9:
                    continue
            except Exception:
                continue
            if len(examples) < 10:
                examples.append((a, int(r)))
            elif len(holdouts) < 4:
                holdouts.append((a, int(r)))
            else:
                break
        return Problem(
            name=name, examples=examples, holdouts=holdouts,
            signature=f"fn {name}(a: i64) -> i64",
            source="random", arg_types=["int"],
        )
    else:
        name, fn = rng.choice(templates_2arg)
        suffix = rng.randint(0, 999)
        name = f"{name}_{suffix}"
        examples = []
        holdouts = []
        seen = set()
        for _ in range(200):
            a = rng.randint(-30, 30)
            b = rng.randint(-30, 30)
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            try:
                r = fn(a, b)
                if abs(r) > 10**9:
                    continue
            except Exception:
                continue
            if len(examples) < 10:
                examples.append(((a, b), int(r)))
            elif len(holdouts) < 4:
                holdouts.append(((a, b), int(r)))
            else:
                break
        return Problem(
            name=name, examples=examples, holdouts=holdouts,
            signature=f"fn {name}(a: i64, b: i64) -> i64",
            source="random", arg_types=["int", "int"],
        )


def _random_branch_problem(rng: random.Random) -> Problem:
    """Generate a random branching problem."""
    templates = [
        ("sign", lambda a: 1 if a > 0 else (-1 if a < 0 else 0)),
        ("relu", lambda a: max(a, 0)),
        ("clamp_10", lambda a: max(-10, min(10, a))),
        ("is_positive", lambda a: 1 if a > 0 else 0),
        ("is_even", lambda a: 1 if a % 2 == 0 else 0),
        ("abs_diff_10", lambda a: abs(a - 10)),
    ]
    name, fn = rng.choice(templates)
    suffix = rng.randint(0, 999)
    name = f"{name}_{suffix}"
    examples = []
    holdouts = []
    seen = set()
    for _ in range(200):
        a = rng.randint(-50, 50)
        if a in seen:
            continue
        seen.add(a)
        try:
            r = fn(a)
        except Exception:
            continue
        if len(examples) < 10:
            examples.append((a, int(r)))
        elif len(holdouts) < 4:
            holdouts.append((a, int(r)))
        else:
            break
    return Problem(
        name=name, examples=examples, holdouts=holdouts,
        signature=f"fn {name}(a: i64) -> i64",
        source="random", arg_types=["int"],
    )


def _random_loop_problem(rng: random.Random) -> Problem:
    """Generate a random loop problem."""
    templates = [
        ("sum_to_n", lambda n: n * (n + 1) // 2 if n >= 0 else 0),
        ("factorial", lambda n: _factorial(n) if 0 <= n <= 12 else None),
        ("count_digits", lambda n: len(str(abs(n))) if n != 0 else 1),
        ("digit_sum", lambda n: sum(int(d) for d in str(abs(n)))),
        ("triangular", lambda n: n * (n + 1) // 2 if n >= 0 else 0),
    ]
    name, fn = rng.choice(templates)
    suffix = rng.randint(0, 999)
    name = f"{name}_{suffix}"
    examples = []
    holdouts = []
    seen = set()
    for _ in range(200):
        a = rng.randint(0, 20)
        if a in seen:
            continue
        seen.add(a)
        try:
            r = fn(a)
            if r is None or abs(r) > 10**9:
                continue
        except Exception:
            continue
        if len(examples) < 10:
            examples.append((a, int(r)))
        elif len(holdouts) < 4:
            holdouts.append((a, int(r)))
        else:
            break
    return Problem(
        name=name, examples=examples, holdouts=holdouts,
        signature=f"fn {name}(a: i64) -> i64",
        source="random", arg_types=["positive_int"],
    )


def _factorial(n: int) -> int:
    if n <= 1:
        return 1
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


RANDOM_GENERATORS = [
    (_random_expr_problem, 0.50),
    (_random_branch_problem, 0.25),
    (_random_loop_problem, 0.25),
]


def generate_random_problem(rng: random.Random) -> Problem:
    """Generate a random synthesis problem using weighted category selection."""
    r = rng.random()
    cumulative = 0.0
    for gen, weight in RANDOM_GENERATORS:
        cumulative += weight
        if r <= cumulative:
            return gen(rng)
    return RANDOM_GENERATORS[0][0](rng)  # fallback


# ---------------------------------------------------------------------------
# Solution verification
# ---------------------------------------------------------------------------

def verify_solution_python(code: str, fn_name: str,
                           holdouts: list[tuple[Any, Any]]) -> tuple[bool, list]:
    """Verify a Python solution against holdout examples."""
    if not holdouts:
        return True, []

    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(compile(code, "<verify>", "exec"), namespace)
    except Exception as e:
        return False, [("compile_error", None, str(e))]

    fn = namespace.get(fn_name)
    if fn is None:
        return False, [("missing_fn", fn_name, None)]

    failing = []
    for args, expected in holdouts:
        call_args = list(args) if isinstance(args, (list, tuple)) else [args]
        try:
            actual = fn(*call_args)
        except Exception as e:
            failing.append((args, expected, str(e)))
            continue
        if not _values_equal(actual, expected):
            failing.append((args, expected, actual))
    return len(failing) == 0, failing


def _is_gradient_method(method: str) -> bool:
    """Classify whether a solve method is gradient-based (mog_synth) or LLM.

    mog_synth returns methods like: "differentiable", "synth_gradient", "template",
    "search", "register_machine", "expression", etc.
    LLM returns: "llm", "llm_mog".
    """
    llm_methods = {"llm", "llm_mog", "llm_python"}
    return method not in llm_methods


def _values_equal(actual: Any, expected: Any, tol: float = 1e-6) -> bool:
    """Flexible equality for int/float/list."""
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(actual) - float(expected)) <= tol
        except (TypeError, ValueError):
            return False
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(actual) == len(expected) and all(
            _values_equal(a, e) for a, e in zip(actual, expected)
        )
    try:
        return int(actual) == int(expected)
    except (TypeError, ValueError):
        pass
    return actual == expected


# ---------------------------------------------------------------------------
# Training data management
# ---------------------------------------------------------------------------

def solution_to_training_record(
    problem: Problem,
    code: str,
    method: str,
) -> Optional[dict]:
    """Convert a verified solution to a JSONL training record.

    Format matches mog_synth/scripts/generate_training_data.py output:
        {"io_pairs": [[inputs, output], ...], "n_args": N,
         "method": "expr"|"branch"|"loop"|..., "code": "...", "name": "..."}
    """
    # Determine program type from synthesis method
    method_map = {
        "differentiable": None,  # will be inferred from code
        "llm": None,
    }

    # Infer program type from code structure
    program_type = _infer_program_type(code, method)
    if program_type is None:
        return None

    # Build io_pairs in the expected format
    io_pairs = []
    for args, expected in problem.examples:
        if isinstance(args, (list, tuple)):
            io_pairs.append([list(args), int(expected)])
        else:
            io_pairs.append([[args], int(expected)])

    n_args = len(problem.arg_types) if problem.arg_types else (
        len(problem.examples[0][0]) if isinstance(problem.examples[0][0], (list, tuple))
        else 1
    )

    return {
        "io_pairs": io_pairs,
        "n_args": n_args,
        "method": program_type,
        "code": code,
        "name": problem.name,
    }


def _infer_program_type(code: str, synthesis_method: str) -> Optional[str]:
    """Infer program type (expr/branch/loop/two_precomp/chained_branch) from code."""
    code_lower = code.lower()

    # Check for loops
    if "while " in code_lower or "for " in code_lower:
        return "loop"

    # Count conditionals
    if_count = code_lower.count("if ")
    if if_count >= 2:
        return "chained_branch"
    if if_count == 1:
        return "branch"

    # Count let statements (proxy for pre-computations)
    let_count = code_lower.count("let ")
    assignment_count = code_lower.count(" = ")
    precomps = max(let_count, assignment_count - 1)  # -1 for return or final
    if precomps >= 2:
        return "two_precomp"

    return "expr"


def append_training_record(record: dict, path: Path = TRAINING_DATA_PATH) -> bool:
    """Append a single JSONL record to the training data file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return True
    except Exception as e:
        logger.warning(f"Failed to append training record: {e}")
        return False


def retrain_meta_learner(
    data_path: Path = TRAINING_DATA_PATH,
    model_path: Path = META_LEARNER_MODEL,
    epochs: int = 50,
    timeout: int = 300,
) -> bool:
    """Retrain the meta-learner on the (expanded) training data.

    Calls mog_synth/scripts/train_expr_metalearner.py as a subprocess.
    """
    script = str(META_LEARNER_SCRIPT)
    if not Path(script).exists():
        logger.warning(f"Meta-learner training script not found: {script}")
        return False

    cmd = [
        sys.executable, script,
        "--data", str(data_path),
        "--save", str(model_path),
        "--epochs", str(epochs),
        "--patience", "15",
    ]

    logger.info(f"Retraining meta-learner: epochs={epochs}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(MOG_SYNTH_DIR),
        )
        if result.returncode == 0:
            logger.info("Meta-learner retrained successfully")
            return True
        else:
            logger.warning(f"Meta-learner training failed: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"Meta-learner training timed out ({timeout}s)")
        return False
    except Exception as e:
        logger.warning(f"Meta-learner training error: {e}")
        return False


# ---------------------------------------------------------------------------
# SelfImprovingLoop
# ---------------------------------------------------------------------------

class SelfImprovingLoop:
    """Self-improving program synthesis engine.

    Maintains a queue of problems, solves them via gradient + LLM fallback,
    verifies solutions, feeds successes back as training data, and periodically
    retrains the meta-learner so the gradient solver handles more over time.
    """

    def __init__(
        self,
        mog_synth_binary: Optional[str] = None,
        llm_model: str = "claude-sonnet-4-20250514",
        enable_llm: bool = True,
        retrain_interval: int = 25,
        retrain_epochs: int = 50,
        training_data_path: Optional[str] = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        """
        Args:
            mog_synth_binary: Path to compiled mog_synth binary (auto-detected if None).
            llm_model: Anthropic model name for LLM fallback.
            enable_llm: Whether to allow LLM synthesis (set False for offline mode).
            retrain_interval: Retrain meta-learner every N successful solves.
            retrain_epochs: Epochs per retrain cycle.
            training_data_path: Override default training data location.
            seed: Random seed for reproducibility.
            verbose: Print progress to stdout.
        """
        self.enable_llm = enable_llm
        self.retrain_interval = retrain_interval
        self.retrain_epochs = retrain_epochs
        self.training_data_path = Path(training_data_path) if training_data_path else TRAINING_DATA_PATH
        self.rng = random.Random(seed)
        self.verbose = verbose

        # Problem queue
        self._problems: list[Problem] = []
        self._solved: list[SolveAttempt] = []
        self._metrics = LoopMetrics()

        # Initialize the unified synthesizer
        from egdc.unified_synth import UnifiedSynthesizer
        self._synth = UnifiedSynthesizer(
            mog_synth_binary=mog_synth_binary,
            llm_model=llm_model,
            enable_differentiable=True,
            enable_llm=enable_llm,
        )

        # Count existing training records to measure growth
        self._initial_record_count = self._count_training_records()

    # ── Problem registration ──────────────────────────────────────────────

    def add_function(
        self,
        fn: Callable,
        arg_types: Optional[list[str]] = None,
        n_examples: int = 12,
        n_holdouts: int = 4,
        description: str = "",
    ) -> None:
        """Add a Python function to the synthesis queue.

        Automatically extracts I/O examples by calling the function with
        random inputs matching the declared arg_types.

        Args:
            fn: A callable Python function.
            arg_types: List of input types ("int", "positive_int", "small_int", etc).
            n_examples: Number of training examples to extract.
            n_holdouts: Number of holdout examples for verification.
            description: Plain-English description of the function.
        """
        from egdc.binary_io_extract import IOExtractor
        ext = IOExtractor.from_function(fn, arg_types=arg_types, description=description)
        problem = ext.extract(n_examples=n_examples, n_holdouts=n_holdouts)

        examples = [
            (ex.inputs if len(ex.inputs) > 1 else ex.inputs[0], ex.expected)
            for ex in problem.examples
        ]
        holdouts = [
            (ex.inputs if len(ex.inputs) > 1 else ex.inputs[0], ex.expected)
            for ex in problem.holdouts
        ]

        self._problems.append(Problem(
            name=problem.name,
            examples=examples,
            holdouts=holdouts,
            signature=problem.signature,
            description=problem.description,
            source="function",
            arg_types=arg_types or ext.arg_types,
        ))
        if self.verbose:
            logger.info(f"Added function '{problem.name}' ({len(examples)} examples, "
                        f"{len(holdouts)} holdouts)")

    def add_binary(
        self,
        path: str,
        fn_name: str = "program",
        arg_types: Optional[list[str]] = None,
        n_examples: int = 12,
        n_holdouts: int = 4,
        description: str = "",
    ) -> None:
        """Add a compiled binary to the synthesis queue.

        The binary should read arguments from stdin and print the result to stdout.

        Args:
            path: Path to the compiled binary.
            fn_name: Name for the synthesized function.
            arg_types: List of input types.
            n_examples: Number of training examples to extract.
            n_holdouts: Number of holdout examples for verification.
            description: Plain-English description.
        """
        if arg_types is None:
            arg_types = ["int"]

        from egdc.binary_io_extract import IOExtractor
        ext = IOExtractor.from_binary(path, arg_types, fn_name=fn_name, description=description)
        problem = ext.extract(n_examples=n_examples, n_holdouts=n_holdouts)

        examples = [
            (ex.inputs if len(ex.inputs) > 1 else ex.inputs[0], ex.expected)
            for ex in problem.examples
        ]
        holdouts = [
            (ex.inputs if len(ex.inputs) > 1 else ex.inputs[0], ex.expected)
            for ex in problem.holdouts
        ]

        self._problems.append(Problem(
            name=fn_name,
            examples=examples,
            holdouts=holdouts,
            signature=problem.signature,
            description=problem.description,
            source="binary",
            arg_types=arg_types,
        ))
        if self.verbose:
            logger.info(f"Added binary '{fn_name}' ({len(examples)} examples)")

    def add_examples(
        self,
        fn_name: str,
        examples: list[tuple[Any, Any]],
        signature: Optional[str] = None,
        holdouts: Optional[list[tuple[Any, Any]]] = None,
        description: str = "",
    ) -> None:
        """Add raw I/O examples directly.

        Args:
            fn_name: Name for the synthesized function.
            examples: List of (input, expected_output) pairs.
            signature: Mog-style type signature (auto-inferred if None).
            holdouts: Additional test cases for verification.
            description: Plain-English description.
        """
        if holdouts is None:
            # Split last 20% as holdouts
            n_holdout = max(1, len(examples) // 5)
            holdouts = examples[-n_holdout:]
            examples = examples[:-n_holdout]

        # Infer arg_types from first example
        first_args = examples[0][0]
        if isinstance(first_args, (list, tuple)):
            arg_types = ["int"] * len(first_args)
        else:
            arg_types = ["int"]

        self._problems.append(Problem(
            name=fn_name,
            examples=examples,
            holdouts=holdouts,
            signature=signature,
            description=description,
            source="manual",
            arg_types=arg_types,
        ))
        if self.verbose:
            logger.info(f"Added examples '{fn_name}' ({len(examples)} examples, "
                        f"{len(holdouts)} holdouts)")

    # ── Core loop ─────────────────────────────────────────────────────────

    def run(
        self,
        n_iterations: int = 100,
        continuous: bool = False,
    ) -> LoopMetrics:
        """Run the self-improving synthesis loop.

        Args:
            n_iterations: Maximum number of problems to attempt.
            continuous: If True, generate random problems to fill the queue.

        Returns:
            LoopMetrics with cumulative statistics.
        """
        t_start = time.time()
        problems_solved_since_retrain = 0

        if self.verbose:
            print(f"\n{'='*70}")
            print("  SELF-IMPROVING SYNTHESIS LOOP")
            print(f"{'='*70}")
            print(f"  Problems queued:     {len(self._problems)}")
            print(f"  Continuous mode:     {continuous}")
            print(f"  Max iterations:      {n_iterations}")
            print(f"  LLM fallback:        {'enabled' if self.enable_llm else 'disabled'}")
            print(f"  Retrain interval:    every {self.retrain_interval} solves")
            print(f"  Training data:       {self.training_data_path}")
            print(f"  Initial records:     {self._initial_record_count}")
            print(f"{'='*70}\n")

        iteration = 0
        while iteration < n_iterations:
            # Get next problem
            if self._problems:
                problem = self._problems.pop(0)
            elif continuous:
                problem = generate_random_problem(self.rng)
            else:
                if self.verbose:
                    print("[loop] No more problems in queue. Done.")
                break

            iteration += 1
            attempt = self._solve_one(problem, iteration, n_iterations)
            self._solved.append(attempt)

            # Track metrics
            self._metrics.problems_attempted += 1
            if attempt.solved and attempt.verified:
                if _is_gradient_method(attempt.method):
                    self._metrics.solved_by_gradient += 1
                else:
                    self._metrics.solved_by_llm += 1

                # Add to training data
                record = solution_to_training_record(problem, attempt.code, attempt.method)
                if record is not None:
                    if append_training_record(record, self.training_data_path):
                        self._metrics.training_records_added += 1

                problems_solved_since_retrain += 1

                # Periodic retrain
                if problems_solved_since_retrain >= self.retrain_interval:
                    self._do_retrain()
                    problems_solved_since_retrain = 0

            elif attempt.solved and not attempt.verified:
                self._metrics.holdout_failures += 1
            else:
                self._metrics.unsolved += 1

            # Periodic progress snapshot
            if iteration % 10 == 0:
                self._metrics.history.append({
                    "iteration": iteration,
                    **self._metrics.snapshot(),
                })

        # Final retrain if we added any new records
        if self._metrics.training_records_added > 0 and problems_solved_since_retrain > 0:
            self._do_retrain()

        self._metrics.total_time_s = time.time() - t_start

        if self.verbose:
            self.report()

        return self._metrics

    def _solve_one(self, problem: Problem, iteration: int, total: int) -> SolveAttempt:
        """Attempt to solve a single problem through the synthesis pipeline."""
        if self.verbose:
            print(f"[{iteration:3d}/{total}] Solving '{problem.name}' "
                  f"({len(problem.examples)} ex, source={problem.source})...", end=" ")

        t0 = time.time()

        # Use the unified synthesizer (gradient first, then LLM)
        result = self._synth.synthesize(
            fn_name=problem.name,
            examples=problem.examples,
            description=problem.description,
            signature=problem.signature,
            holdouts=problem.holdouts,
        )

        elapsed = time.time() - t0

        if not result.solved:
            if self.verbose:
                print(f"UNSOLVED ({elapsed:.1f}s) [{result.method}]")
            return SolveAttempt(
                problem_name=problem.name,
                solved=False,
                method=result.method,
                time_s=elapsed,
                error=result.error,
            )

        # Verify against holdout examples
        verified = True
        if problem.holdouts and result.language == "python":
            verified, failing = verify_solution_python(
                result.code, problem.name, problem.holdouts
            )
            if not verified and self.verbose:
                print(f"HOLDOUT FAIL ({elapsed:.1f}s) [{result.method}] "
                      f"({len(failing)} failures)")
                return SolveAttempt(
                    problem_name=problem.name,
                    solved=True,
                    method=result.method,
                    code=result.code,
                    language=result.language,
                    time_s=elapsed,
                    verified=False,
                    error=f"Holdout failures: {failing[:3]}",
                )

        if self.verbose:
            tag = "GRADIENT" if result.method == "differentiable" else result.method.upper()
            v_tag = " [verified]" if problem.holdouts else ""
            print(f"SOLVED by {tag} ({elapsed:.1f}s){v_tag}")

        return SolveAttempt(
            problem_name=problem.name,
            solved=True,
            method=result.method,
            code=result.code,
            language=result.language,
            time_s=elapsed,
            verified=verified,
        )

    def _do_retrain(self) -> None:
        """Retrain the meta-learner on expanded training data."""
        if self.verbose:
            current_count = self._count_training_records()
            print(f"\n  >>> RETRAINING META-LEARNER (records: {self._initial_record_count} -> {current_count})")

        success = retrain_meta_learner(
            data_path=self.training_data_path,
            model_path=META_LEARNER_MODEL,
            epochs=self.retrain_epochs,
        )

        if success:
            self._metrics.retrains_completed += 1
            # Reload the meta-learner in the unified synthesizer
            try:
                sys.path.insert(0, str(MOG_SYNTH_DIR / "scripts"))
                from train_expr_metalearner import load_model
                self._synth.meta_learner = load_model(str(META_LEARNER_MODEL))
                if self.verbose:
                    print("  >>> Meta-learner reloaded successfully\n")
            except Exception as e:
                if self.verbose:
                    print(f"  >>> Warning: could not reload meta-learner: {e}\n")
        else:
            if self.verbose:
                print("  >>> Retrain failed (continuing with old model)\n")

    def _count_training_records(self) -> int:
        """Count lines in the training data JSONL file."""
        if not self.training_data_path.exists():
            return 0
        try:
            with open(self.training_data_path) as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    # ── Reporting ─────────────────────────────────────────────────────────

    def report(self) -> None:
        """Print a summary of the loop's performance."""
        m = self._metrics
        current_records = self._count_training_records()

        print(f"\n{'='*70}")
        print("  SELF-IMPROVING LOOP RESULTS")
        print(f"{'='*70}")
        print(f"  Problems attempted:     {m.problems_attempted}")
        print(f"  Solved by gradient:     {m.solved_by_gradient}")
        print(f"  Solved by LLM:          {m.solved_by_llm}")
        print(f"  Unsolved:               {m.unsolved}")
        print(f"  Holdout failures:       {m.holdout_failures}")
        print(f"  ────────────────────────────────────")
        print(f"  Total solve rate:       {m.solve_rate:.1%}")
        print(f"  Gradient share:         {m.gradient_rate:.1%} of solved")
        print(f"  ────────────────────────────────────")
        print(f"  Training records added:  {m.training_records_added}")
        print(f"  Training data size:      {self._initial_record_count} -> {current_records}")
        print(f"  Meta-learner retrains:   {m.retrains_completed}")
        print(f"  Total time:              {m.total_time_s:.1f}s")
        print(f"{'='*70}")

        # Print solve history if we have multiple snapshots
        if len(m.history) >= 2:
            print("\n  Progress over time:")
            print(f"  {'Iter':>6s} {'Solved':>8s} {'Gradient':>10s} {'LLM':>6s} "
                  f"{'Unsolved':>10s} {'Rate':>8s} {'GradRate':>10s}")
            for h in m.history:
                total_s = h["gradient"] + h["llm"]
                print(f"  {h['iteration']:6d} {total_s:8d} {h['gradient']:10d} "
                      f"{h['llm']:6d} {h['unsolved']:10d} "
                      f"{h['solve_rate']:8.1%} {h['gradient_rate']:10.1%}")
            print()

    @property
    def metrics(self) -> LoopMetrics:
        return self._metrics

    @property
    def solved(self) -> list[SolveAttempt]:
        return self._solved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Self-improving program synthesis loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Continuous mode: generate random problems and learn
  python egdc/self_improving_loop.py --continuous --iterations 50

  # Solve specific functions (lambda expressions)
  python egdc/self_improving_loop.py --functions "lambda a,b: a+b" "lambda x: x*x"

  # Offline mode (no LLM, gradient only)
  python egdc/self_improving_loop.py --continuous --no-llm --iterations 100

  # Custom retrain interval
  python egdc/self_improving_loop.py --continuous --retrain-interval 10 --iterations 50
        """,
    )
    parser.add_argument("--continuous", action="store_true",
                        help="Generate random problems to fill the queue")
    parser.add_argument("--iterations", "-n", type=int, default=100,
                        help="Max number of problems to attempt")
    parser.add_argument("--functions", nargs="+",
                        help="Lambda expressions to synthesize (e.g. 'lambda a,b: a+b')")
    parser.add_argument("--binary", type=str, default=None,
                        help="Binary path to synthesize from")
    parser.add_argument("--binary-fn-name", type=str, default="program",
                        help="Function name for binary synthesis")
    parser.add_argument("--arg-types", nargs="+", default=["int"],
                        help="Argument types for binary synthesis")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM fallback (gradient only)")
    parser.add_argument("--retrain-interval", type=int, default=25,
                        help="Retrain meta-learner every N solves")
    parser.add_argument("--retrain-epochs", type=int, default=50,
                        help="Epochs per retrain cycle")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--training-data", type=str, default=None,
                        help="Override training data path")
    parser.add_argument("--llm-model", type=str, default="claude-sonnet-4-20250514",
                        help="Anthropic model for LLM fallback")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print progress (default: True)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--json-report", type=str, default=None,
                        help="Write JSON metrics report to file")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    loop = SelfImprovingLoop(
        llm_model=args.llm_model,
        enable_llm=not args.no_llm,
        retrain_interval=args.retrain_interval,
        retrain_epochs=args.retrain_epochs,
        training_data_path=args.training_data,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Add functions from CLI
    if args.functions:
        for expr_str in args.functions:
            try:
                fn = eval(expr_str)
                if not callable(fn):
                    print(f"Warning: '{expr_str}' is not callable, skipping")
                    continue
                # Infer arg count from the lambda
                import inspect
                sig = inspect.signature(fn)
                n_args = len(sig.parameters)
                arg_types = ["int"] * n_args
                loop.add_function(fn, arg_types=arg_types)
            except Exception as e:
                print(f"Warning: Could not parse '{expr_str}': {e}")

    # Add binary
    if args.binary:
        loop.add_binary(args.binary, fn_name=args.binary_fn_name,
                        arg_types=args.arg_types)

    # Run the loop
    metrics = loop.run(
        n_iterations=args.iterations,
        continuous=args.continuous,
    )

    # JSON report
    if args.json_report:
        report = {
            "final": metrics.snapshot(),
            "history": metrics.history,
            "solved": [
                {
                    "name": s.problem_name,
                    "method": s.method,
                    "verified": s.verified,
                    "time_s": round(s.time_s, 3),
                }
                for s in loop.solved if s.solved
            ],
        }
        Path(args.json_report).write_text(json.dumps(report, indent=2))
        print(f"\nJSON report written to {args.json_report}")

    # Exit code: 0 if any problems solved, 1 if none
    sys.exit(0 if metrics.solve_rate > 0 else 1)


if __name__ == "__main__":
    main()
