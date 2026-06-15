"""The six MCP tools: NL → I/O pairs → verified program → actual output.

Every heavy piece is reused, not reimplemented:

- prompt → I/O pairs: ``ncpu.autoresearch.prompt_parser`` (arrow notation,
  doctests, asserts, "returns" prose).
- examples → verified program: ``ncpu.synthesis_api.server`` —
  ``handle_synthesize_request`` shells out to the Rust binary
  (``mog_synth --problem-json -``) and transpiles via ``--transpile``.
- bank stats: ``ncpu.synthesis_api.server.read_bank_stats``.
- library lookup: ``ncpu.mcp_server.fingerprint`` mirrors the Rust
  solved-cache fingerprint so hits are answered without a subprocess.
- candidate verification / execution: ``ncpu.mcp_server.sandbox`` runs
  client-drafted code in a subprocess sandbox against the same examples
  (tools 5 and 6, re-exported here).

The honest-refusal contract: when the synthesizer cannot find a program
that reproduces *every* example, the tool returns ``verified: false``
with the backend's reason — plus the full out-of-domain protocol: the
client should draft the function itself and submit it through
``verify_candidate`` with the same examples (echoed in the refusal
payload), so the only code ever shown to the user is example-verified.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional

from ncpu.autoresearch.prompt_parser import (
    _ARROW_RE,
    _DOCTEST_RE,
    _RETURNS_RE,
    extract_from_prompt,
)
from ncpu.autoresearch.types import IoPair
from ncpu.mcp_server.fingerprint import examples_fingerprint, lookup_solved
from ncpu.mcp_server.sandbox import run_program, verify_candidate
from ncpu.synthesis_api.server import (
    MAX_TIMEOUT_S,
    SynthConfig,
    handle_synthesize_request,
    read_bank_stats,
)

_LANGUAGES = ("python", "rust", "typescript", "go")

_GUIDANCE = (
    "Provide concrete input/output examples like: f(2,3) -> 5. "
    "Arrow notation (fn(args) -> out), doctests (>>> fn(args)), and "
    "asserts (assert fn(args) == out) are all understood. nsynth solves "
    "over int, [int], and str inputs with int outputs. If synthesis "
    "still refuses once examples exist, draft the function yourself and "
    "submit it through verify_candidate with those same examples; only "
    "verified code should be shown to the user."
)

# The full out-of-domain protocol, attached to every synthesis refusal
# together with the examples (so the client never has to re-parse them).
_REFUSAL_PROTOCOL = (
    "synthesis refused; draft the function yourself and submit it "
    "through verify_candidate with these same examples; only verified "
    "code should be shown to the user. Use run_program to execute the "
    "verified function on new inputs."
)


def _refusal(reason: str, examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Honest refusal carrying the full cascade protocol + the examples."""
    return {
        "verified": False,
        "reason": reason,
        "guidance": _REFUSAL_PROTOCOL,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Tool 1: synthesize_from_examples
# ---------------------------------------------------------------------------


def synthesize_from_examples(
    config: SynthConfig,
    name: str,
    examples: list[dict[str, Any]],
    language: str = "python",
    timeout_s: Optional[float] = None,
) -> dict[str, Any]:
    """Examples in, proof-carrying code out — or an honest refusal."""
    if language not in _LANGUAGES:
        return _refusal(
            f"unsupported language {language!r} "
            f"(expected one of {', '.join(_LANGUAGES)})",
            examples,
        )

    request: dict[str, Any] = {"name": name, "examples": examples}
    if timeout_s is not None:
        request["timeout_s"] = min(float(timeout_s), MAX_TIMEOUT_S)

    status, body = handle_synthesize_request(request, config)
    if status != 200:
        # Malformed input (400) or backend unavailable (503).
        return _refusal(str(body.get("error", "bad request")), examples)

    if not body.get("success"):
        reason = body.get("error") or body.get("method") or "no program found"
        return _refusal(str(reason), examples)

    transpiled = body.get("transpiled") or {}
    code = transpiled.get(language)
    result: dict[str, Any] = {
        "verified": True,
        "method": body.get("method"),
        "mog": body.get("code"),
        "code": code,
        "language": language,
        "examples_checked": len(examples),
        "elapsed_ms": body.get("elapsed_ms"),
    }
    if code is None:
        result["transpile_warning"] = (
            f"transpile to {language} failed; the verified Mog source in "
            "'mog' is the source of truth"
        )
    return result


# ---------------------------------------------------------------------------
# Tool 2: synthesize_from_prompt
# ---------------------------------------------------------------------------


def _infer_function_name(prompt: str) -> Optional[str]:
    """Most-called identifier across the example patterns in the prompt."""
    names: Counter[str] = Counter()
    for regex in (_ARROW_RE, _DOCTEST_RE, _RETURNS_RE):
        for match in regex.finditer(prompt):
            names[match.group(1)] += 1
    if not names:
        return None
    return names.most_common(1)[0][0]


def _io_pair_to_example(pair: IoPair) -> Optional[dict[str, Any]]:
    """IoPair → nsynth example, or None if not representable.

    nsynth solves over int / [int] / str inputs and int outputs;
    keyword arguments have no positional slot in a Mog signature.
    """
    if pair.kwargs:
        return None
    if isinstance(pair.expected, bool) or not isinstance(pair.expected, int):
        return None
    inputs: list[Any] = []
    for arg in pair.args:
        if isinstance(arg, bool):
            return None
        if isinstance(arg, int) or isinstance(arg, str):
            inputs.append(arg)
        elif isinstance(arg, list) and all(
            isinstance(x, int) and not isinstance(x, bool) for x in arg
        ):
            inputs.append(arg)
        else:
            return None
    if not inputs:
        return None
    return {"inputs": inputs, "expected": pair.expected}


def synthesize_from_prompt(
    config: SynthConfig,
    prompt: str,
    language: str = "python",
    timeout_s: Optional[float] = None,
) -> dict[str, Any]:
    """NL prompt → extracted I/O pairs (echoed back) → tool-1 synthesis."""
    report = extract_from_prompt(prompt)
    entry_point = report.entry_point or _infer_function_name(prompt)
    if entry_point is not None and report.entry_point is None:
        # Re-run with the inferred name so pairs for *other* identifiers
        # mentioned in prose don't pollute the example set.
        report = extract_from_prompt(prompt, entry_point=entry_point)

    if not report.io_pairs:
        return {
            "verified": False,
            "reason": "no I/O examples found",
            "guidance": _GUIDANCE,
            "function_name": entry_point,
        }

    examples: list[dict[str, Any]] = []
    skipped = 0
    for pair in report.io_pairs:
        ex = _io_pair_to_example(pair)
        if ex is None:
            skipped += 1
        else:
            examples.append(ex)

    extracted = [
        {"args": p.args, "kwargs": p.kwargs, "expected": p.expected}
        for p in report.io_pairs
    ]

    if not examples:
        # Out of the synthesizer's domain, but not out of the cascade's:
        # echo the pairs in verify_candidate-ready form so the client can
        # draft code and push it through the same verification gate.
        verify_ready = [
            {"inputs": p.args, "expected": p.expected}
            for p in report.io_pairs
            if not p.kwargs
        ]
        return {
            "verified": False,
            "reason": (
                "examples found but none are representable: nsynth solves "
                "over int, [int], and str inputs with int outputs"
            ),
            "guidance": _REFUSAL_PROTOCOL,
            "examples": verify_ready,
            "function_name": entry_point,
            "extracted_examples": extracted,
        }

    result = synthesize_from_examples(
        config,
        name=entry_point or "synthesized",
        examples=examples,
        language=language,
        timeout_s=timeout_s,
    )
    result["function_name"] = entry_point
    result["extracted_examples"] = extracted
    result["extraction_sources"] = report.sources
    if skipped:
        result["examples_skipped"] = skipped
    return result


# ---------------------------------------------------------------------------
# Tool 3: consult_library
# ---------------------------------------------------------------------------


def consult_library(
    config: SynthConfig, examples: list[dict[str, Any]]
) -> dict[str, Any]:
    """Exact examples-fingerprint lookup against the solved bank.

    A hit means the synthesizer already verified a program against these
    exact examples in some past session — return it instantly.
    """
    try:
        for i, ex in enumerate(examples):
            if not isinstance(ex, dict) or "inputs" not in ex or "expected" not in ex:
                raise ValueError(
                    f"examples[{i}] must be an object with 'inputs' and 'expected'"
                )
        fp = examples_fingerprint(examples)
    except (ValueError, TypeError, KeyError) as exc:
        return {"hit": False, "error": f"bad examples: {exc}"}

    record = lookup_solved(config.bank_path("solved"), fp)
    if record is None:
        return {"hit": False, "fingerprint": fp}
    return {
        "hit": True,
        "fingerprint": fp,
        "method": record["method"],
        "mog": record["code"],
        "success_count": record["success_count"],
        "last_used_at": record["last_used_at"],
    }


# ---------------------------------------------------------------------------
# Tool 4: library_stats
# ---------------------------------------------------------------------------


def library_stats(config: SynthConfig) -> dict[str, Any]:
    """Sizes of the three persistent memory banks (observable learning)."""
    return read_bank_stats(config)


__all__ = [
    "synthesize_from_examples",
    "synthesize_from_prompt",
    "consult_library",
    "library_stats",
    "verify_candidate",
    "run_program",
]
