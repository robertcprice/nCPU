"""LLM-resample solver — re-run the target LLM with a bigger sampling budget.

When the baseline/agent-runner gave up after 2 attempts at fixed temps,
this solver tries a wider sweep: multiple independent samples at each
of several temperatures, extracting the function body from each and
returning the first candidate that passes the original test suite.

Integrated with the compounding stack: if a model + tokenizer is passed
in, we also allow the coprocessor gate to flip between baseline and
NPCoT-sampled settings across attempts. So the LLM-resample stage is
strictly more powerful than what ran at eval time, because it explores
more of the sampling distribution.

This solver never lives as a stub — it needs a model handle, so callers
construct it via :func:`make_llm_resampler` and install it via
:attr:`CascadeConfig.extra_solvers`.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

from ncpu.autoresearch.types import WorkItem


def make_llm_resampler(
    *,
    generate_fn: Callable[[str, float, int], str],
    extract_code_fn: Callable[[str, str], str],
    temperatures: Sequence[float] = (0.3, 0.5, 0.7, 0.9),
    samples_per_temp: int = 4,
    verify_fn: Optional[Callable[[WorkItem, str], tuple[bool, str]]] = None,
):
    """Return a solver callable the cascade can install.

    Parameters
    ----------
    generate_fn : Callable
        ``generate_fn(prompt, temperature, max_new_tokens) -> str`` — one
        model sample. Wrap your model handle + tokenizer here.
    extract_code_fn : Callable
        ``extract_code_fn(raw_output, prompt) -> str`` — normalize the
        model's output into a body suitable for appending to ``prompt``.
        Use :func:`ncpu.self_optimizing.humaneval_runner._extract_code`.
    temperatures : sequence
        Sampling temperatures to sweep over.
    samples_per_temp : int
        Samples drawn per temperature. Total samples = len(temps) * this.
    verify_fn : callable, optional
        Per-sample verifier (bool, detail) = fn(item, code). If omitted,
        falls back to :func:`cascade.verify_python_solution`. Useful if
        the caller wants to skip verification (e.g. "any output is OK").

    Returns
    -------
    callable
        A ``(item, *, budget_seconds) -> Optional[str]`` solver.
    """
    from ncpu.autoresearch.cascade import verify_python_solution as _default_verify
    verify = verify_fn or _default_verify

    def _solver(item: WorkItem, *, budget_seconds: float = 120.0) -> Optional[str]:
        t0 = time.perf_counter()
        for temp in temperatures:
            for _ in range(samples_per_temp):
                if time.perf_counter() - t0 > budget_seconds:
                    return None
                try:
                    raw = generate_fn(item.prompt, float(temp), 400)
                except Exception:
                    continue
                code = extract_code_fn(raw, item.prompt)
                passed, _detail = verify(item, code)
                if passed:
                    return code
        return None

    _solver.__name__ = "llm_resample"
    _solver.__doc__ = f"LLM resampler: temps={list(temperatures)}, n={samples_per_temp}"
    return _solver
