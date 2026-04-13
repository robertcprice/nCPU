"""HumanEval dataset and evaluation for Python code diffusion.

Downloads/loads the OpenAI HumanEval benchmark (164 Python programming problems)
and provides a PyTorch Dataset for training + an evaluation harness for pass@k.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from egdc.python.tokenizer import (
    PythonCodeTokenizer,
    MASK_TOKEN,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    VOCAB_SIZE,
)


# ---------------------------------------------------------------------------
# HumanEval data loading
# ---------------------------------------------------------------------------

HUMANEVAL_URL = (
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
)
CACHE_DIR = Path.home() / ".cache" / "egdc" / "humaneval"


def _download_humaneval(force: bool = False) -> Path:
    """Download HumanEval dataset if not cached. Returns path to .jsonl.gz."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = CACHE_DIR / "HumanEval.jsonl.gz"
    if gz_path.exists() and not force:
        return gz_path

    print(f"Downloading HumanEval from {HUMANEVAL_URL} ...")
    import urllib.request
    urllib.request.urlretrieve(HUMANEVAL_URL, gz_path)
    print(f"Saved to {gz_path}")
    return gz_path


def load_humaneval(force_download: bool = False) -> List[Dict[str, Any]]:
    """Load HumanEval problems as a list of dicts.

    Each dict has keys:
        task_id, prompt, entry_point, canonical_solution, test
    """
    gz_path = _download_humaneval(force=force_download)
    problems = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class HumanEvalDataset(Dataset):
    """PyTorch Dataset for training diffusion on HumanEval canonical solutions.

    Each item returns:
        masked_tokens:   (seq_len,) int64 - solution with random positions masked
        mask_positions:  (seq_len,) bool  - True where tokens are masked
        original_tokens: (seq_len,) int64 - original unmasked solution
        spec_tokens:     (spec_len,) int64 - encoded prompt (function sig + docstring)
        timestep:        scalar float     - diffusion timestep in [0, 1]
    """

    def __init__(
        self,
        seq_len: int = 512,
        spec_len: int = 256,
        seed: int = 42,
        num_diffusion_steps: int = 1000,
        problems: Optional[List[Dict[str, Any]]] = None,
        repeat: int = 1,
    ):
        """Initialize dataset.

        Args:
            seq_len: Fixed length for solution (completion) tokens.
            spec_len: Fixed length for spec (prompt) tokens.
            seed: Random seed.
            num_diffusion_steps: Number of discrete diffusion timesteps.
            problems: Pre-loaded problems list, or None to download.
            repeat: How many times to repeat the dataset (for more training data).
        """
        super().__init__()
        self.tokenizer = PythonCodeTokenizer()
        self.seq_len = seq_len
        self.spec_len = spec_len
        self.num_diffusion_steps = num_diffusion_steps
        self.rng = random.Random(seed)

        if problems is None:
            problems = load_humaneval()

        # Build training data: (prompt_tokens, solution_tokens) pairs
        self.data: List[Tuple[List[int], List[int]]] = []
        for prob in problems:
            prompt = prob.get("prompt", "")
            solution = prob.get("canonical_solution", "")
            if not solution.strip():
                continue

            prompt_toks = self.tokenizer.encode(prompt, add_bos_eos=False)
            solution_toks = self.tokenizer.encode(solution, add_bos_eos=True)

            self.data.append((prompt_toks, solution_toks))

        # Repeat dataset for more training data
        if repeat > 1:
            self.data = self.data * repeat

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        prompt_toks, solution_toks = self.data[idx]

        # --- Solution tokens: pad/truncate to seq_len ---
        prog = self.tokenizer.pad(list(solution_toks), self.seq_len)
        original_tokens = torch.tensor(prog, dtype=torch.long)

        # --- Spec tokens: pad/truncate prompt to spec_len ---
        spec = self.tokenizer.pad(list(prompt_toks), self.spec_len)
        spec_tokens = torch.tensor(spec, dtype=torch.long)

        # --- Diffusion masking ---
        t_int = self.rng.randint(0, self.num_diffusion_steps - 1)
        timestep = torch.tensor(t_int / self.num_diffusion_steps, dtype=torch.float32)
        mask_prob = t_int / self.num_diffusion_steps

        # Create mask: only mask actual code tokens (not PAD, BOS, EOS)
        mask_positions = torch.zeros(self.seq_len, dtype=torch.bool)
        for i in range(self.seq_len):
            tok = prog[i]
            if tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                continue
            if self.rng.random() < mask_prob:
                mask_positions[i] = True

        # Apply mask
        masked_tokens = original_tokens.clone()
        masked_tokens[mask_positions] = MASK_TOKEN

        return masked_tokens, mask_positions, original_tokens, spec_tokens, timestep


# ---------------------------------------------------------------------------
# Evaluation: pass@k
# ---------------------------------------------------------------------------


def _estimate_pass_at_k(
    num_samples: int, num_correct: int, k: int
) -> float:
    """Unbiased estimator for pass@k (from the HumanEval paper).

    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.prod(
        (num_samples - num_correct - i) / (num_samples - i) for i in range(k)
    )


def _safe_execute(code: str, timeout: int = 5) -> Tuple[bool, str]:
    """Execute Python code in a sandboxed subprocess.

    Returns (success, output_or_error).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", "/tmp"),
                "PYTHONPATH": "",
            },
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _run_problem_tests(
    prompt: str, completion: str, test_code: str, entry_point: str
) -> bool:
    """Assemble and run a HumanEval problem's tests.

    Constructs: prompt + completion + test harness, then executes.
    Returns True if all tests pass.
    """
    # The test code typically defines a check() function that calls the entry_point
    full_code = prompt + completion + "\n" + test_code + f"\ncheck({entry_point})\n"
    success, output = _safe_execute(full_code, timeout=10)
    return success


@torch.no_grad()
def evaluate_humaneval(
    model: torch.nn.Module,
    tokenizer: PythonCodeTokenizer,
    generate_fn,
    num_problems: Optional[int] = None,
    k: int = 1,
    num_samples_per_problem: int = 1,
    temperature: float = 0.8,
    seq_len: int = 512,
    spec_len: int = 256,
) -> Dict[str, Any]:
    """Evaluate a model on HumanEval.

    Args:
        model: The diffusion model.
        tokenizer: PythonCodeTokenizer instance.
        generate_fn: Callable(model, spec_tokens, seq_len, ...) -> (1, L) token IDs.
        num_problems: How many problems to evaluate (None = all 164).
        k: k for pass@k metric.
        num_samples_per_problem: Number of completions to generate per problem.
        temperature: Sampling temperature.
        seq_len: Max sequence length for generation.
        spec_len: Max spec length.

    Returns:
        Dict with pass_at_k, per_problem results, etc.
    """
    problems = load_humaneval()
    if num_problems is not None:
        problems = problems[:num_problems]

    device = next(model.parameters()).device
    model.eval()

    results = []
    num_correct = 0
    total = 0

    for prob in problems:
        task_id = prob["task_id"]
        prompt = prob["prompt"]
        test_code = prob["test"]
        entry_point = prob["entry_point"]

        # Encode prompt as spec tokens
        prompt_toks = tokenizer.encode(prompt, add_bos_eos=False)
        prompt_toks = tokenizer.pad(prompt_toks, spec_len)
        spec_tokens = torch.tensor([prompt_toks], dtype=torch.long, device=device)

        problem_correct = 0

        for sample_idx in range(num_samples_per_problem):
            # Generate completion
            generated = generate_fn(
                model=model,
                spec_tokens=spec_tokens,
                seq_len=seq_len,
                temperature=temperature,
            )

            # Decode generated tokens to code
            gen_ids = generated[0].tolist()
            completion = tokenizer.decode(gen_ids, skip_special=True)

            # Test the completion
            passed = _run_problem_tests(prompt, completion, test_code, entry_point)
            if passed:
                problem_correct += 1

        pass_k = _estimate_pass_at_k(num_samples_per_problem, problem_correct, k)
        results.append({
            "task_id": task_id,
            "num_samples": num_samples_per_problem,
            "num_correct": problem_correct,
            f"pass@{k}": pass_k,
        })

        num_correct += problem_correct
        total += num_samples_per_problem

        print(
            f"  {task_id}: {problem_correct}/{num_samples_per_problem} passed, "
            f"pass@{k}={pass_k:.3f}"
        )

    # Aggregate
    avg_pass_k = sum(r[f"pass@{k}"] for r in results) / max(len(results), 1)

    summary = {
        "num_problems": len(results),
        "k": k,
        "num_samples_per_problem": num_samples_per_problem,
        f"pass@{k}": avg_pass_k,
        "total_correct": num_correct,
        "total_samples": total,
        "per_problem": results,
    }

    print(f"\n=== HumanEval Results ===")
    print(f"  Problems: {len(results)}")
    print(f"  pass@{k}: {avg_pass_k:.4f}")
    print(f"  Total correct: {num_correct}/{total}")

    return summary
