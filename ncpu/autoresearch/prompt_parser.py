"""Extract tests from free-form user prompts → WorkItem.

The cascade, runner, and compounding store all consume WorkItems whose
``io_pairs`` list was populated by the miner (parsing a benchmark's
hidden test suite). For real user prompts — "Write me a function to
reverse a string" with maybe an example or two — there is no hidden
test suite. This module bridges the gap: it takes a natural-language
prompt + optional code stub and emits a :class:`WorkItem` with a
best-effort extracted :class:`IoPair` list.

Supported extraction patterns (roughly in order of specificity):

1. **Explicit asserts**: ``assert reverse("hello") == "olleh"`` anywhere
   in the prompt, including inside fenced code blocks.
2. **Doctests**: ``>>> reverse("hello")`` on one line, ``'olleh'`` on
   the next. Standard Python docstring convention.
3. **Arrow notation**: ``reverse("hello") -> "olleh"`` or with unicode
   ``→``. Tolerates surrounding prose.
4. **"returns" prose**: ``reverse("hello") returns "olleh"``. Looser
   heuristic — last resort before falling back.
5. **Entry-point inference**: if the prompt contains a ``def foo(...):``
   stub, ``foo`` is used as the entry point. Otherwise the caller must
   supply it, or the parser guesses from the most-called identifier in
   the extracted expressions.

What this parser *does not* do:

* It does not synthesize tests when the prompt contains no examples.
  For that path, a caller (``ExtendedCodingAssistant`` in
  ``coding_assistant.py``) asks the target LLM to propose candidate
  test cases, then passes them here for structural parsing. The parser
  itself is deterministic and has no LLM dependency.
* It does not execute any input to resolve references. All values must
  be Python literals at the syntactic level (numbers, strings, lists,
  tuples, dicts, True/False/None). Non-literal expressions are skipped.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ncpu.autoresearch.types import IoPair, WorkItem


# ----------------------------------------------------------------------
# Regex primitives
# ----------------------------------------------------------------------

_DEF_RE = re.compile(r"def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?\s*:")

# Find fenced code blocks so we can search both body and fence.
_FENCE_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)

# Doctest: >>> f(args)  (one line)  then optional output on next line.
_DOCTEST_RE = re.compile(
    r">>> *(\w+)\((.*?)\)\s*$\n(?!\s*(?:>>>|\.\.\.))\s*(.+)$",
    re.MULTILINE,
)

# Arrow notation: fname(args) -> out  or fname(args) → out
_ARROW_RE = re.compile(
    r"(\w+)\s*\(([^)]*)\)\s*(?:->|→)\s*([^\n,;]+)"
)

# "returns" prose: fname(args) returns out
_RETURNS_RE = re.compile(
    r"(\w+)\s*\(([^)]*)\)\s*returns?\s+([^\n,;]+)",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------

def _parse_literal(source: str) -> Optional[Any]:
    """Parse a literal, returning ``None`` if not parseable.

    Tolerates trailing sentence punctuation (``. ! ?``) and inline
    comments introduced by ``#``.
    """
    source = source.strip()
    # Try raw first.
    try:
        return ast.literal_eval(source)
    except Exception:
        pass
    # Strip trailing sentence punctuation and retry.
    trimmed = source.rstrip(" .!?;,")
    if trimmed != source:
        try:
            return ast.literal_eval(trimmed)
        except Exception:
            pass
    # Strip anything after ``#`` (doctest comment) and retry.
    if "#" in trimmed:
        trimmed2 = trimmed.split("#", 1)[0].strip().rstrip(" .!?;,")
        try:
            return ast.literal_eval(trimmed2)
        except Exception:
            pass
    return None


def _parse_call_args(args_src: str) -> Optional[tuple[list, dict]]:
    """Parse positional + keyword args as if from a call expression.

    Accepts e.g. ``1, 2, x=3`` and returns ([1, 2], {"x": 3}) or
    ``None`` if any arg isn't a literal.
    """
    try:
        tree = ast.parse(f"_({args_src})", mode="eval")
    except SyntaxError:
        return None
    call = tree.body
    if not isinstance(call, ast.Call):
        return None
    try:
        args = [ast.literal_eval(a) for a in call.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}
    except (ValueError, SyntaxError):
        return None
    return args, kwargs


def extract_entry_point(prompt: str) -> Optional[str]:
    """Best-effort: pick the function name from a ``def`` line in the prompt."""
    matches = list(_DEF_RE.finditer(prompt))
    if matches:
        # Prefer the LAST def — in case the prompt shows signature after prose.
        return matches[-1].group(1)
    return None


def _extract_doctest_pairs(text: str, entry_point: Optional[str]) -> list[IoPair]:
    """Parse ``>>> fn(args)\\n<output>`` doctest lines."""
    out: list[IoPair] = []
    for match in _DOCTEST_RE.finditer(text):
        name, args_src, output_src = match.group(1), match.group(2), match.group(3)
        if entry_point is not None and name != entry_point:
            continue
        parsed = _parse_call_args(args_src)
        if parsed is None:
            continue
        expected = _parse_literal(output_src)
        if expected is None:
            continue
        out.append(IoPair(args=parsed[0], kwargs=parsed[1], expected=expected))
    return out


def _extract_assert_pairs(text: str, entry_point: Optional[str]) -> list[IoPair]:
    """Parse ``assert fn(args) == expected`` anywhere in the text."""
    from ncpu.autoresearch.miner import extract_io_pairs
    if entry_point is None:
        return []
    return extract_io_pairs(text, entry_point)


def _extract_arrow_pairs(text: str, entry_point: Optional[str]) -> list[IoPair]:
    """Parse ``fn(args) -> out`` arrow-notation pairs."""
    out: list[IoPair] = []
    for match in _ARROW_RE.finditer(text):
        name, args_src, output_src = match.group(1), match.group(2), match.group(3)
        if entry_point is not None and name != entry_point:
            continue
        parsed = _parse_call_args(args_src)
        if parsed is None:
            continue
        expected = _parse_literal(output_src)
        if expected is None:
            continue
        out.append(IoPair(args=parsed[0], kwargs=parsed[1], expected=expected))
    return out


def _split_sentences(text: str) -> list[str]:
    """Very crude sentence splitter: ``. ``, ``! ``, ``? `` or newline."""
    # Preserve newlines so multiline asserts stay intact.
    parts: list[str] = []
    buf = ""
    i = 0
    while i < len(text):
        c = text[i]
        buf += c
        if c in ".!?":
            # Look ahead: end-sentence if followed by whitespace.
            if i + 1 >= len(text) or text[i + 1].isspace():
                parts.append(buf)
                buf = ""
        elif c == "\n":
            parts.append(buf)
            buf = ""
        i += 1
    if buf:
        parts.append(buf)
    return [p.strip() for p in parts if p.strip()]


def _extract_returns_pairs(text: str, entry_point: Optional[str]) -> list[IoPair]:
    """Parse ``fn(args) returns out`` prose — one match per sentence."""
    out: list[IoPair] = []
    for sentence in _split_sentences(text):
        match = _RETURNS_RE.search(sentence)
        if match is None:
            continue
        name, args_src, output_src = match.group(1), match.group(2), match.group(3)
        if entry_point is not None and name != entry_point:
            continue
        parsed = _parse_call_args(args_src)
        if parsed is None:
            continue
        expected = _parse_literal(output_src)
        if expected is None:
            continue
        out.append(IoPair(args=parsed[0], kwargs=parsed[1], expected=expected))
    return out


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

@dataclass
class ExtractionReport:
    """Detail what the parser found (useful when nothing works)."""

    entry_point: Optional[str]
    io_pairs: list[IoPair]
    sources: dict[str, int] = field(default_factory=dict)
    """Count of I/O pairs extracted per pattern (doctest, assert, arrow, returns)."""
    raw_prompt: str = ""

    def ok(self) -> bool:
        return bool(self.io_pairs) and self.entry_point is not None

    def summary(self) -> dict[str, Any]:
        return {
            "entry_point": self.entry_point,
            "io_pairs": len(self.io_pairs),
            "sources": self.sources,
            "ok": self.ok(),
        }


def extract_from_prompt(
    prompt: str,
    *,
    entry_point: Optional[str] = None,
) -> ExtractionReport:
    """Extract a best-effort :class:`ExtractionReport` from a free-form prompt.

    Order of operations:
    1. Determine entry_point (explicit arg > def-line inference > None).
    2. Run each extractor, accumulating unique (args, kwargs, expected)
       triples. Duplicate pairs are dropped so a doctest + a restated
       arrow don't double-count.
    """
    ep = entry_point or extract_entry_point(prompt)
    seen: set[tuple] = set()
    accum: list[IoPair] = []
    sources: dict[str, int] = {}

    def _accept(p: IoPair, tag: str) -> None:
        key = _pair_key(p)
        if key in seen:
            return
        seen.add(key)
        accum.append(p)
        sources[tag] = sources.get(tag, 0) + 1

    for tag, extractor in (
        ("doctest", _extract_doctest_pairs),
        ("assert", _extract_assert_pairs),
        ("arrow", _extract_arrow_pairs),
        ("returns", _extract_returns_pairs),
    ):
        for p in extractor(prompt, ep):
            _accept(p, tag)

    # Also scan fenced code blocks for asserts (which often live there).
    for fence in _FENCE_RE.finditer(prompt):
        body = fence.group(1)
        for p in _extract_assert_pairs(body, ep):
            _accept(p, "assert_fenced")

    return ExtractionReport(
        entry_point=ep, io_pairs=accum, sources=sources, raw_prompt=prompt,
    )


def _freeze(value: Any) -> Any:
    """Recursive freeze of a literal so it's hashable for dedupe."""
    if isinstance(value, list):
        return ("list", tuple(_freeze(v) for v in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((_freeze(k), _freeze(v)) for k, v in value.items())))
    if isinstance(value, set):
        return ("set", tuple(sorted(_freeze(v) for v in value)))
    return value


def _pair_key(p: IoPair) -> tuple:
    """Hashable dedupe key for an IoPair.

    ``_freeze`` must be applied to *every* component — args and kwarg
    values as well as ``expected`` — otherwise list-valued args (e.g.
    ``sum_list([1, 2, 3]) -> 6``) produce an unhashable tuple and the
    ``key in seen`` check raises ``TypeError``.
    """
    return (
        tuple(_freeze(a) for a in p.args),
        tuple(sorted((k, _freeze(v)) for k, v in p.kwargs.items())),
        _freeze(p.expected),
    )


def build_work_item(
    prompt: str,
    *,
    task_id: str = "user/0",
    entry_point: Optional[str] = None,
    extra_io_pairs: Optional[list[IoPair]] = None,
    synth_prompt_template: Optional[str] = None,
) -> Optional[WorkItem]:
    """Turn a user prompt into a :class:`WorkItem` the cascade can consume.

    Returns ``None`` if no entry point can be inferred — without a
    function name we don't know what to solve for.

    ``extra_io_pairs`` is appended after the extracted ones — useful
    when the caller has LLM-synthesized test candidates and wants them
    concatenated with the parser's output.
    """
    report = extract_from_prompt(prompt, entry_point=entry_point)
    if report.entry_point is None:
        return None

    all_pairs = list(report.io_pairs)
    if extra_io_pairs:
        seen: set[tuple] = {_pair_key(p) for p in all_pairs}
        for p in extra_io_pairs:
            key = _pair_key(p)
            if key not in seen:
                seen.add(key)
                all_pairs.append(p)

    # Synthesize a minimal `def check(candidate):` harness so the
    # cascade's verify_python_solution can run the same path it uses
    # for HumanEval/MBPP.
    test_body_lines = ["def check(candidate):"]
    for p in all_pairs:
        args_repr = ", ".join(repr(a) for a in p.args)
        if p.kwargs:
            kwargs_repr = ", ".join(f"{k}={v!r}" for k, v in p.kwargs.items())
            call_repr = f"candidate({args_repr}, {kwargs_repr})"
        else:
            call_repr = f"candidate({args_repr})"
        test_body_lines.append(f"    assert {call_repr} == {p.expected!r}")
    if len(test_body_lines) == 1:
        test_body_lines.append("    pass")  # no pairs: vacuous harness
    test_source = "\n".join(test_body_lines) + "\n"

    # Reconstruct a clean runnable prompt: extract just the def
    # signature of the target entry point and generate a minimal
    # docstring from whatever prose surrounded it. This avoids
    # mixing arrow-notation examples into the Python file.
    def_match = None
    for m in _DEF_RE.finditer(prompt):
        if m.group(1) == report.entry_point:
            def_match = m
            break
    if def_match is not None:
        params = def_match.group(2).strip()
        # Use everything before the def line as the "instruction."
        preamble = prompt[: def_match.start()].strip()
        doc_src = preamble if preamble else f"Implement {report.entry_point}."
        # Collapse to single-line docstring.
        doc_src = " ".join(doc_src.split())[:400]
        runtime_prompt = (
            f"def {report.entry_point}({params}):\n"
            f"    \"\"\"{doc_src}\"\"\"\n"
        )
    else:
        runtime_prompt = (
            f"def {report.entry_point}(*args, **kwargs):\n"
            f"    \"\"\"{' '.join(prompt.split())[:400]}\"\"\"\n"
        )

    return WorkItem(
        task_id=task_id,
        source_benchmark="user",
        prompt=runtime_prompt,
        entry_point=report.entry_point,
        test_source=test_source,
        io_pairs=all_pairs,
        priority=1.0 + 0.1 * len(all_pairs),
        provenance={
            "extraction_sources": report.sources,
            "raw_user_prompt": prompt,
        },
    )
