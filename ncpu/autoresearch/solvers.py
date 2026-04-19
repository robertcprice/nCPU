"""Concrete solver callables plugged into the cascade.

Each solver is a function ``(item, *, budget_seconds) -> Optional[str]``
that returns candidate Python source or ``None``. Solvers are expected to
fail fast: they should never block longer than ``budget_seconds``, and
returning ``None`` is the normal "I don't have a solution" signal.

Currently shipped:

* ``template_match`` — search a small library of Python templates (sum,
  max, count-predicate, filter-predicate, sort-and-pick) for one that
  matches the work item's extracted I/O pairs. Purely CPU, no GPU, no
  network. Cheap first line of defense for array-reduction-shaped
  problems that slipped past the coprocessor library.
* ``nsynth_fast`` — placeholder stub, returns ``None``. Integrating
  nsynth requires translating :class:`IoPair` into its native
  ``Problem`` format, which is being done lazily.
* ``llm_resample`` — placeholder stub. Real implementations are injected
  at runtime via ``CascadeConfig.extra_solvers`` (they need model +
  tokenizer handles that can't live at import time).
"""

from __future__ import annotations

import re
import time
from typing import Callable, Optional

from ncpu.autoresearch.types import IoPair, WorkItem


SolverFn = Callable[[WorkItem], Optional[str]]
"""Signature: ``fn(item, *, budget_seconds) -> Optional[str]``."""


# ----------------------------------------------------------------------
# template_match
# ----------------------------------------------------------------------

_DEF_RE = re.compile(r"def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?\s*:")


def _parse_params(prompt: str, entry_point: str) -> list[str]:
    """Extract positional parameter names from the prompt's def line."""
    for m in _DEF_RE.finditer(prompt):
        if m.group(1) == entry_point:
            raw = m.group(2)
            names: list[str] = []
            for tok in raw.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                # strip type annotation and default
                tok = tok.split(":", 1)[0].strip()
                tok = tok.split("=", 1)[0].strip()
                if tok.startswith(("*", "**")):
                    continue
                names.append(tok)
            return names
    return []


def _apply(fn: Callable, pair: IoPair) -> tuple[bool, object]:
    try:
        return True, fn(*pair.args, **pair.kwargs)
    except Exception:
        return False, None


def _matches_all(fn: Callable, pairs: list[IoPair]) -> bool:
    for p in pairs:
        ok, got = _apply(fn, p)
        if not ok or got != p.expected:
            return False
    return True


# Templates indexed by (arity, first_arg_is_iterable).
# Each entry is (template_code_with_ARG_placeholders, predicate_fn).
# ARG0/ARG1 in the template are replaced with the real param names.
_ARR1_TEMPLATES: list[tuple[str, Callable]] = [
    ("    return sum(ARG0)\n",                          lambda xs: sum(xs)),
    ("    return max(ARG0)\n",                          lambda xs: max(xs)),
    ("    return min(ARG0)\n",                          lambda xs: min(xs)),
    ("    return len(ARG0)\n",                          lambda xs: len(xs)),
    ("    return sum(x for x in ARG0 if x > 0)\n",      lambda xs: sum(x for x in xs if x > 0)),
    ("    return sum(x for x in ARG0 if x % 2 == 0)\n", lambda xs: sum(x for x in xs if x % 2 == 0)),
    ("    return sum(x for x in ARG0 if x % 2 != 0)\n", lambda xs: sum(x for x in xs if x % 2 != 0)),
    ("    return sum(x*x for x in ARG0)\n",             lambda xs: sum(x*x for x in xs)),
    ("    return sorted(ARG0)\n",                       lambda xs: sorted(xs)),
    ("    return sorted(ARG0, reverse=True)\n",         lambda xs: sorted(xs, reverse=True)),
    ("    return list(reversed(ARG0))\n",               lambda xs: list(reversed(xs))),
    ("    return list(dict.fromkeys(ARG0))\n",          lambda xs: list(dict.fromkeys(xs))),
    ("    return [x for x in ARG0 if x > 0]\n",         lambda xs: [x for x in xs if x > 0]),
    ("    return [x for x in ARG0 if x % 2 == 0]\n",    lambda xs: [x for x in xs if x % 2 == 0]),
    ("    return [x*x for x in ARG0]\n",                lambda xs: [x*x for x in xs]),
    ("    return [-x for x in ARG0]\n",                 lambda xs: [-x for x in xs]),
]

_SCALAR1_TEMPLATES: list[tuple[str, Callable]] = [
    ("    return ARG0 * ARG0\n",               lambda x: x * x),
    ("    return ARG0 ** 3\n",                 lambda x: x ** 3),
    ("    return abs(ARG0)\n",                 lambda x: abs(x)),
    ("    return ARG0 + 1\n",                  lambda x: x + 1),
    ("    return ARG0 - 1\n",                  lambda x: x - 1),
    ("    return ARG0 * 2\n",                  lambda x: x * 2),
]

_SCALAR2_TEMPLATES: list[tuple[str, Callable]] = [
    ("    return ARG0 + ARG1\n",               lambda a, b: a + b),
    ("    return ARG0 - ARG1\n",               lambda a, b: a - b),
    ("    return ARG0 * ARG1\n",               lambda a, b: a * b),
    ("    return ARG0 ** ARG1\n",              lambda a, b: a ** b),
    ("    return max(ARG0, ARG1)\n",           lambda a, b: max(a, b)),
    ("    return min(ARG0, ARG1)\n",           lambda a, b: min(a, b)),
    ("    return abs(ARG0 - ARG1)\n",          lambda a, b: abs(a - b)),
]


def _render_body(template: str, param_names: list[str]) -> str:
    """Replace ARG0/ARG1 in the template with real parameter names."""
    out = template
    for i, name in enumerate(param_names):
        out = out.replace(f"ARG{i}", name)
    return out


def _is_iterable_first(pair: IoPair) -> bool:
    return bool(pair.args) and isinstance(pair.args[0], (list, tuple))


def _is_scalar_args(pair: IoPair, arity: int) -> bool:
    return (
        not pair.kwargs
        and len(pair.args) == arity
        and all(isinstance(a, (int, float)) for a in pair.args)
    )


def template_match(item: WorkItem, *, budget_seconds: float = 5.0) -> Optional[str]:
    """Brute-force search over a small Python template library."""
    t0 = time.perf_counter()
    pairs = item.io_pairs
    if not pairs:
        return None

    params = _parse_params(item.prompt, item.entry_point)
    if not params:
        return None
    first = pairs[0]

    candidates: list[tuple[str, Callable]] = []
    if len(first.args) == 1 and _is_iterable_first(first) and not first.kwargs:
        candidates = _ARR1_TEMPLATES
    elif _is_scalar_args(first, 2):
        candidates = _SCALAR2_TEMPLATES
    elif _is_scalar_args(first, 1):
        candidates = _SCALAR1_TEMPLATES

    for tmpl, fn in candidates:
        if time.perf_counter() - t0 > budget_seconds:
            return None
        if _matches_all(fn, pairs):
            return _render_body(tmpl, params)
    return None


# ----------------------------------------------------------------------
# placeholder stubs — to be overridden at runtime
# ----------------------------------------------------------------------

def nsynth_fast(item: WorkItem, *, budget_seconds: float = 15.0) -> Optional[str]:
    """Placeholder: nsynth integration pending.

    Returning ``None`` means the cascade skips this stage. When nsynth
    gains a "synthesize from literal I/O pairs" CLI, this stub will call
    it via subprocess and translate its 5-tuple program back to Python.
    """
    return None


def llm_resample_stub(item: WorkItem, *, budget_seconds: float = 60.0) -> Optional[str]:
    """Placeholder: caller must override via ``CascadeConfig.extra_solvers``."""
    return None


def llm_teacher_stub(item: WorkItem, *, budget_seconds: float = 120.0) -> Optional[str]:
    """Placeholder: stronger-model API integration not yet wired."""
    return None


SOLVER_FUNCTIONS: dict[str, SolverFn] = {
    "template_match": template_match,
    "nsynth_fast": nsynth_fast,
    "llm_resample": llm_resample_stub,
    "llm_teacher": llm_teacher_stub,
}
